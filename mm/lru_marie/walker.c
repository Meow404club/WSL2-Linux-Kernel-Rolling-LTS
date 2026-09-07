// SPDX-License-Identifier: GPL-2.0
/*
 * mm/lru_marie/walker.c — Marie's SIMD-accelerated PTE walker.
 *
 * The walker is Marie's hot signal harvester. Per pgdat,
 * rate-limited via marie_walker_interval() (HZ/30..HZ depending on
 * pressure), marie_walk_pgdat() snapshots the running mm_struct's,
 * walks each page table to PMD granularity, and at every PMD invokes
 * lru_marie_simd_young_pte_mask() to extract the young-bit bitmap
 * of the entire 512-PTE page in one SIMD pass (AVX-512F / AVX2 / SSE2
 * on x86; scalar fallback on arm64 and other arches via the generic
 * variant -- NEON acceleration is a future optimisation, see
 * mm/lru_marie/simd_generic.c).
 * The FPU bracket wraps only the SIMD scan of each PMD (a single-shot
 * begin+scan+end), so it never spans PMDs or bloom misses and the heavy
 * per-young-PTE work stays outside it. For each PTE flagged young the
 * walker clears its accessed bit under the ptl and records the PFN; once
 * the ptl is dropped it promotes each folio straight to the head gen via
 * marie_state_inc_tier() on the per-PFN byte, lock-free and
 * preempt-enabled -- no promote queue, no pass-end drain.
 *
 * A per-pgdat bloom filter (marie_bloom_*) feeds back from
 * lru_marie_look_around() (rmap-side, called from
 * folio_referenced_one()) to the walker: rmap flags PMDs whose target
 * folio was young, the walker reads that bitmap and skips PMDs the
 * rmap path has not flagged. The bloom is double-buffered (active /
 * inactive) and rotated at pass end so the walker reads the feedback
 * accumulated during the previous reclaim window.
 *
 * Bloom is the *only* coupling between rmap and the walker.
 * lru_marie_look_around() does NOT promote (no tier++, no
 * PG_referenced) on the surrounding folios; the walker handles tier++
 * via young-bit detection on bloom-hit PMDs. This split keeps the
 * rmap path PTL-bounded and lock-free, while the walker pays the
 * SIMD scan + tier++ cost only for hot PMDs.
 */

#define pr_fmt(fmt) "lru_marie: " fmt

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/cleanup.h>
#include <linux/hash.h>
#include <linux/list.h>
#include <linux/lru_marie.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>
#include <linux/mmu_notifier.h>
#include <linux/mmzone.h>
#include <linux/pagewalk.h>
#include <linux/percpu.h>
#include <linux/prefetch.h>
#include <linux/printk.h>
#include <linux/rcupdate.h>
#include <linux/rmap.h>
#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/xarray.h>

#include "../internal.h"	/* folio_pte_batch_flags / FPB_MERGE_YOUNG_DIRTY */
#include "walker_compat.h"	/* marie_folio_pte_batch, lazy_mmu / young-ptes shims */
#include "state.h"
#include "simd.h"

/*
 * ---------------------------------------------------------------------
 *  look-around (rmap-side opportunistic PMD scan)
 * ---------------------------------------------------------------------
 *
 * lru_marie_look_around() is called from rmap.c::folio_referenced_one() while
 * the rmap caller already holds the page table lock for the target
 * folio's PTE.  We piggyback on that PTL to scan up to
 * MARIE_LOOK_AROUND_BATCH PTEs of the surrounding PMD and clear young bits
 * found there in batch — what would otherwise cost one rmap walk per
 * neighbouring folio amortises into a single PMD pass, and subsequent
 * folio_referenced() calls on those folios get a more accurate "young
 * since last reclaim cycle" answer.
 *
 * Crucially, we do NOT call folio_set_referenced() on the surrounding
 * folios.  Doing so would cascade into a reclaim-side promote and
 * starve reclaim under fault-heavy workloads (memhog, browser tab
 * churn) where every recently-faulted PTE has its young bit set —
 * see the comment above the test_and_clear loop below.
 *
 * Returns true iff the target folio's own PTE(s) were young.  That's the
 * value folio_referenced_one() folds into its referenced count, exactly
 * mirroring what test_and_clear_young_ptes_notify() would have returned
 * from the bare clear_flush_young_ptes_notify branch.
 *
 * Lock contract: caller holds the PTL and (via rmap_walk) one of the
 * anon_vma / i_mmap rwsems. We DO NOT take any Marie lock here.
 * Promotion of folios with an external hotness signal happens out-of-
 * band: folio_mark_accessed -> lru_marie_mark_accessed calls
 * marie_state_inc_tier, which promotes straight to the head gen via a
 * synchronous marie_state_move_to_gen on the same path -- a lock-free
 * CAS. No new lock-ordering relationship between rmap and Marie state
 * is introduced.
 */
#define MARIE_LOOK_AROUND_BATCH BITS_PER_LONG	/* PTEs scanned per call */

/*
 * Version-shimmed mm APIs (test_and_clear_young_ptes_notify, lazy_mmu_mode_*,
 * folio_pte_batch) live in walker_compat.h so this file is identical across
 * kernels.
 */

/* Forward decl: bloom Producer used by look_around. Definition lives in
 * the walker helpers section alongside the walker-side Consumer. */
static void marie_bloom_set(int nid, unsigned long pmd_addr);

bool lru_marie_look_around(struct page_vma_mapped_walk *pvmw, unsigned int nr)
{
	pte_t *pte = pvmw->pte;
	unsigned long addr = pvmw->address;
	unsigned long start, end;
	struct vm_area_struct *vma = pvmw->vma;
	struct folio *target = pfn_folio(pvmw->pfn);
	struct mem_cgroup *memcg;
	struct pglist_data *pgdat = folio_pgdat(target);
	int i;

	lockdep_assert_held(pvmw->ptl);

	/*
	 * Clamp the target clear to the PMD (pte-page) boundary. @nr is the
	 * folio's pvmw.nr_pages; for a large folio (THP/mTHP) that straddles a
	 * PMD boundary or is only partially mapped at @addr, clearing @nr
	 * consecutive PTEs forward from @pte would run off the end of this
	 * pte page into the adjacent -- possibly freed (PAGE_POISONING) or
	 * non-existent -- page table, write-faulting in
	 * test_and_clear_young_ptes_notify(). pvmw->pte/ptl only cover this
	 * one page table. (MGLRU's lru_gen_look_around sidesteps this by
	 * clearing a single pte; the surrounding scan below is already
	 * PMD-bounded via @end + marie_folio_pte_batch.) The common
	 * PMD-aligned THP (nr == 512, addr at index 0) is unaffected.
	 */
	{
		unsigned int pmd_room =
			(((addr | ~PMD_MASK) + 1) - addr) >> PAGE_SHIFT;

		if (nr > pmd_room)
			nr = pmd_room;
	}

	/* Always clear the target folio's own young bit and propagate the
	 * result to the caller, regardless of whether we go on to scan the
	 * surrounding PMD. */
	if (!test_and_clear_young_ptes_notify(vma, addr, pte, nr))
		return false;

	/*
	 * Producer: feed the per-pgdat bloom. The target folio was young,
	 * so this PMD has at least one hot PTE worth visiting on the next
	 * walker pass. This is the *only* rmap-side signal Marie gives the
	 * walker -- look_around does NOT promote (no tier++, no
	 * PG_referenced) on the surrounding folios; the walker handles
	 * tier++ via young-bit detection on bloom-hit PMDs.
	 */
	marie_bloom_set(pgdat->node_id, addr & PMD_MASK);

	/* If the PTL is contended skip the surrounding scan — somebody else
	 * is waiting and we shouldn't extend our hold time. */
	if (spin_is_contended(pvmw->ptl))
		return true;

	/* PFN-mapped VMAs don't carry struct folio backings on every PTE;
	 * skip them rather than feed garbage to pfn_folio(). */
	if (vma->vm_flags & VM_SPECIAL)
		return true;

	/* Compute a PMD-bounded surrounding range centred on @addr.  We
	 * scan at most MARIE_LOOK_AROUND_BATCH PTEs and never cross either
	 * the PMD or the VMA boundary. */
	start = max(addr & PMD_MASK, vma->vm_start);
	end = min(addr | ~PMD_MASK, vma->vm_end - 1) + 1;

	if (end - start == PAGE_SIZE)
		return true;

	if (end - start > MARIE_LOOK_AROUND_BATCH * PAGE_SIZE) {
		if (addr - start < MARIE_LOOK_AROUND_BATCH * PAGE_SIZE / 2)
			end = start + MARIE_LOOK_AROUND_BATCH * PAGE_SIZE;
		else if (end - addr < MARIE_LOOK_AROUND_BATCH * PAGE_SIZE / 2)
			start = end - MARIE_LOOK_AROUND_BATCH * PAGE_SIZE;
		else {
			start = addr - MARIE_LOOK_AROUND_BATCH * PAGE_SIZE / 2;
			end = addr + MARIE_LOOK_AROUND_BATCH * PAGE_SIZE / 2;
		}
	}

	memcg = get_mem_cgroup_from_folio(target);

	lazy_mmu_mode_enable();

	pte -= (addr - start) / PAGE_SIZE;

	for (i = 0, addr = start; addr != end;
	     i += nr, pte += nr, addr += nr * PAGE_SIZE) {
		unsigned long pfn;
		pte_t ptent = ptep_get(pte);
		struct folio *folio;

		nr = 1;

		/* Inline minimal get_pte_pfn — vmscan.c's version is
		 * MGLRU-static and we only need a subset of its checks. */
		if (!pte_present(ptent))
			continue;
		if (pte_special(ptent))
			continue;
		pfn = pte_pfn(ptent);
		if (is_zero_pfn(pfn))
			continue;
		if (!pfn_valid(pfn))
			continue;
		if (pfn < pgdat->node_start_pfn || pfn >= pgdat_end_pfn(pgdat))
			continue;

		folio = pfn_folio(pfn);
		if (folio_nid(folio) != pgdat->node_id)
			continue;

		rcu_read_lock();
		if (folio_memcg(folio) != memcg)
			folio = NULL;
		rcu_read_unlock();
		if (!folio)
			continue;

		if (folio_test_large(folio)) {
			const unsigned int max_nr = (end - addr) >> PAGE_SHIFT;

			/* Version-agnostic neighbour batching; see walker_compat.h. */
			nr = marie_folio_pte_batch(folio, addr, pte, ptent, max_nr);
		}

		/* The target folio's young bit was already cleared above and
		 * its referenced status will be re-derived by the caller from
		 * our return value — don't double-clear it here. */
		if (folio == target)
			continue;

		/*
		 * Clear young bits across the surrounding PMD in batch. We
		 * deliberately do NOT touch any tier / PG_referenced state on
		 * the neighbours here: under a fault-heavy allocator (memhog,
		 * browser tab churn) every recently-faulted PTE has its young
		 * bit set, and amplifying that into a hot signal on
		 * ~MARIE_LOOK_AROUND_BATCH neighbours per rmap call cascades
		 * through promote-in-place and starves the reclaim path of
		 * evictable folios. The only signal look_around emits for the
		 * neighbours is the per-pgdat bloom (set above) — that tells
		 * the next walker pass "this PMD had at least one hot PTE",
		 * and the walker itself does per-PTE tier++ from young-bit
		 * detection, preserving per-folio cardinality in the "hot"
		 * signal that drives MARIE_TIER promotions.
		 */
		test_and_clear_young_ptes_notify(vma, addr, pte, nr);
	}

	lazy_mmu_mode_disable();
	mem_cgroup_put(memcg);

	return true;
}
EXPORT_SYMBOL_GPL(lru_marie_look_around);

/*
 * ---------------------------------------------------------------------
 *  Walker helpers: adaptive walker rate, per-pgdat state
 * ---------------------------------------------------------------------
 */

/*
 * Adaptive walker rate. High pressure -> short interval
 * (frequent walks -> fresh tier signal); idle -> long interval (don't
 * burn CPU). Returns jiffies until the next walker pass for this pgdat.
 *
 * Watermarks come from ZONE_NORMAL when present; for builds where
 * ZONE_NORMAL is absent we fall back to the first populated zone.
 *
 * All four stage intervals are runtime-tunable via
 * /sys/kernel/mm/lru_marie/walker_interval_{critical,low,normal,idle}_ms;
 * defaults preserve the original HZ/30, HZ/10, HZ/4, HZ cadence.
 */
static unsigned long marie_walker_interval(struct pglist_data *pgdat)
{
	struct zone *zone = NULL;
	unsigned long free, high, low, min;
	int zid;

	for (zid = 0; zid < MAX_NR_ZONES; zid++) {
		struct zone *z = &pgdat->node_zones[zid];

		if (!populated_zone(z))
			continue;
		if (zid == ZONE_NORMAL) {
			zone = z;
			break;
		}
		if (!zone)
			zone = z;
	}
	if (!zone)
		return READ_ONCE(marie_walker_interval_idle);

	free = sum_zone_node_page_state(pgdat->node_id, NR_FREE_PAGES);
	high = high_wmark_pages(zone);
	low  = low_wmark_pages(zone);
	min  = min_wmark_pages(zone);

	if (free < min)
		return READ_ONCE(marie_walker_interval_critical);
	if (free < low)
		return READ_ONCE(marie_walker_interval_low);
	if (free < high)
		return READ_ONCE(marie_walker_interval_normal);
	return READ_ONCE(marie_walker_interval_idle);
}

/*
 * ---------------------------------------------------------------------
 *  Bloom filter -- rmap → walker forward feedback
 * ---------------------------------------------------------------------
 *
 * Per-pgdat probabilistic set of "PMDs the rmap path saw young in since
 * the last walker pass." Keyed by PMD index (>>PMD_SHIFT), m=1<<15
 * (4 KiB per filter, 8 KiB per pgdat), k=2.
 *
 * Producer: lru_marie_look_around() (rmap-side, runs under PTL during eviction
 *   folio_referenced walks). Sets bits in @inactive.
 *
 * Consumer: marie_walk_pmd_range() (walker hot path, runs under PTL).
 *   Tests bits in @active. Bloom miss -> skip the PMD's SIMD scan.
 *
 * Pass-end (marie_walk_pgdat): swap active/inactive under @lock and clear
 * the new inactive. The walker therefore reads the rmap feedback that
 * accumulated during the previous reclaim window.
 *
 * @warmed_up is the force_scan kill-switch: sticky-true on the first
 * Producer write per pgdat. Until then, the walker bypasses bloom and
 * scans every PMD (covers cold-boot and freshly-online pgdats where
 * rmap has never fed the filter).
 *
 * Lazy alloc with GFP_ATOMIC -- look_around runs under PTL, so any
 * sleeping alloc would deadlock. Allocation failure leaves @inactive
 * NULL; the next look_around call retries. With both bitmaps NULL the
 * walker falls back to force_scan via @warmed_up == false.
 */
#define MARIE_BLOOM_SHIFT		15
#define MARIE_BLOOM_SIZE		(1U << MARIE_BLOOM_SHIFT)	/* 32K bits */

struct marie_bloom {
	spinlock_t	lock;		/* serialises swap + alloc */
	unsigned long	*active;	/* read by walker */
	unsigned long	*inactive;	/* written by look_around */
	bool		warmed_up;	/* sticky: true after first Producer set */
};

static struct marie_bloom marie_blooms[MAX_NUMNODES];

static inline void marie_bloom_keys(unsigned long pmd_addr, int *key)
{
	u32 hash = hash_long(pmd_addr >> PMD_SHIFT, MARIE_BLOOM_SHIFT * 2);

	key[0] = hash & (MARIE_BLOOM_SIZE - 1);
	key[1] = (hash >> MARIE_BLOOM_SHIFT) & (MARIE_BLOOM_SIZE - 1);
}

static unsigned long *marie_bloom_alloc_atomic(void)
{
	return bitmap_zalloc(MARIE_BLOOM_SIZE, GFP_ATOMIC);
}

/*
 * Producer: feed @pmd_addr into pgdat @nid's inactive bloom. Idempotent.
 * Bitops are lock-free; only the lazy-alloc slow path takes b->lock.
 */
static void marie_bloom_set(int nid, unsigned long pmd_addr)
{
	struct marie_bloom *b;
	unsigned long *filter;
	unsigned long flags;
	int key[2];

	if (nid < 0 || nid >= MAX_NUMNODES)
		return;
	b = &marie_blooms[nid];

	marie_bloom_keys(pmd_addr, key);

	filter = READ_ONCE(b->inactive);
	if (filter) {
		if (!test_bit(key[0], filter))
			set_bit(key[0], filter);
		if (!test_bit(key[1], filter))
			set_bit(key[1], filter);
		if (!READ_ONCE(b->warmed_up))
			WRITE_ONCE(b->warmed_up, true);
		return;
	}

	/* Slow path: lazy allocate both bitmaps. */
	spin_lock_irqsave(&b->lock, flags);
	if (!b->inactive)
		b->inactive = marie_bloom_alloc_atomic();
	if (!b->active)
		b->active = marie_bloom_alloc_atomic();
	if (!b->inactive) {
		spin_unlock_irqrestore(&b->lock, flags);
		return;	/* OOM: walker will use force_scan via !warmed_up */
	}
	filter = b->inactive;
	if (!test_bit(key[0], filter))
		set_bit(key[0], filter);
	if (!test_bit(key[1], filter))
		set_bit(key[1], filter);
	b->warmed_up = true;
	spin_unlock_irqrestore(&b->lock, flags);
}

/*
 * Consumer: walker hot path. Returns true iff @pmd_addr is in pgdat
 * @nid's active bloom. NULL active -> false (caller's force_scan path
 * covers it).
 */
static bool marie_bloom_test(int nid, unsigned long pmd_addr)
{
	unsigned long *filter;
	int key[2];

	if (nid < 0 || nid >= MAX_NUMNODES)
		return false;

	filter = READ_ONCE(marie_blooms[nid].active);
	if (!filter)
		return false;

	marie_bloom_keys(pmd_addr, key);
	return test_bit(key[0], filter) && test_bit(key[1], filter);
}

/*
 * Pass-end: swap active <- inactive, clear new inactive. Called from
 * marie_walk_pgdat under no other lock.
 */
static void marie_bloom_swap(int nid)
{
	struct marie_bloom *b;
	unsigned long *tmp;
	unsigned long flags;

	if (nid < 0 || nid >= MAX_NUMNODES)
		return;
	b = &marie_blooms[nid];

	spin_lock_irqsave(&b->lock, flags);
	tmp = b->active;
	b->active = b->inactive;
	b->inactive = tmp;
	if (b->inactive)
		bitmap_zero(b->inactive, MARIE_BLOOM_SIZE);
	spin_unlock_irqrestore(&b->lock, flags);
}

static inline bool marie_bloom_warmed(int nid)
{
	if (nid < 0 || nid >= MAX_NUMNODES)
		return false;
	return READ_ONCE(marie_blooms[nid].warmed_up);
}

/*
 * Per-CPU walk context: a preallocated mm snapshot buffer so the walker
 * doesn't kmalloc inside its hot entry path (the walker can be entered
 * from direct reclaim, where allocator recursion is disallowed).
 *
 * Ownership is established by marie_walker_busy below: the pass owner
 * pins to its CPU via migrate_disable() and claims the per-CPU ctx
 * with this_cpu_cmpxchg(marie_walker_busy, 0, 1). A preempted-and-
 * resumed reclaimer that reaches marie_walk_pgdat on the same CPU will
 * find the flag set and bail, preventing concurrent reuse of the
 * snapshot buffer. The walker pass itself stays preemptible so
 * cond_resched() inside marie_walk_pmd_range remains effective.
 *
 * marie_walker_next[] lives in the walker section below alongside the
 * rest of the walker state.
 */
#define MARIE_WALK_MAX_MMS	256

struct marie_walk_ctx {
	struct mm_struct	*mms[MARIE_WALK_MAX_MMS];
	int			n_mms;
	/*
	 * Deferred inc_tier worklist: young+TRACKED PFNs collected under one
	 * PMD's ptl, drained after the ptl is dropped so the tier bump runs
	 * lock-free and preempt-enabled. Sized to the SIMD bitmap's bit
	 * capacity = the most young PTEs a single PMD scan can report.
	 */
	unsigned long		pfnbuf[MARIE_SIMD_PTE_BITMAP_LONGS * BITS_PER_LONG];
};

static DEFINE_PER_CPU(struct marie_walk_ctx, marie_walker_ctx);
static DEFINE_PER_CPU(unsigned int, marie_walker_busy);

/*
 * ---------------------------------------------------------------------
 *  Walker -- SIMD + adaptive + per-pgdat
 * ---------------------------------------------------------------------
 *
 * Per pgdat, rate-limited via marie_walker_interval (HZ/30 .. HZ).
 * Each PMD scans young bits via lru_marie_simd_young_pte_mask (single-shot
 * begin+scan+end; AVX-512F / AVX2 / SSE2 on x86; scalar fallback on arm64
 * and other arches), so the FPU bracket wraps only the scan. The young
 * PFNs it finds are aged under the ptl and their tier bumps drained after
 * the ptl is dropped (see marie_walk_pmd_range). Cross-node folios are
 * filtered out so each pgdat owns its work cleanly.
 *
 * Walker tier promotion is synchronous: marie_state_inc_tier (on the
 * per-PFN byte) handles both the non-saturated bump and the saturate ->
 * in-place promote inside the same call, so there is no promote queue
 * and no pass-end promote drain.
 *
 * Lock contract:
 *   per-PMD:         holds the existing pte_offset_map_lock ptl
 *   per-PTE body:    lock-free -- marie_state_inc_tier mutates only the
 *                    per-PFN state byte
 *   walker_visits:   lock-free atomic_inc on the global
 *                    marie_gen_walker_visits[gen][type] counter (read as a
 *                    >= 1 boolean; reset in marie_try_advance_head)
 *   bloom rotation:  per-pgdat marie_blooms[nid].lock (irqsave), taken only
 *                    for lazy alloc / pass-end swap
 *   per-pgdat deadline: cmpxchg on marie_walker_next[nid]
 *
 * Lock ordering: the walker takes the pte ptl and, under it, at most the
 * per-pgdat bloom lock (a leaf). It takes NO lru_lock and NO per-type
 * lock anywhere, so it does not participate in -- and cannot invert --
 * Marie's lru_lock -> type_lock hierarchy.
 */

/*
 * Per-pgdat walker deadline (jiffies). One pass per pgdat per
 * marie_walker_interval(pgdat) is allowed; concurrent reclaimers /
 * kswapd cycles atomic-cmpxchg to claim the slot.
 *
 * MARIE_WALK_MAX_MMS bounds the per-pass task snapshot (see the per-CPU
 * marie_walker_ctx definition earlier in this file).
 */
static atomic_long_t marie_walker_next[MAX_NUMNODES];

/*
 * The FPU bracket wraps only the SIMD scan of one PMD (single-shot
 * lru_marie_simd_young_pte_mask = begin+scan+end), so it is opened and
 * closed per PMD and never spans PMDs or bloom-miss stretches. The heavy
 * per-young-PTE work is split in two: the ptep_test_and_clear_young aging
 * runs under the ptl (it must), collecting the young PFNs into a per-cpu
 * worklist; the tier bumps are then drained after the ptl is dropped, so
 * marie_state_inc_tier -- the dominant cost -- runs lock-free and
 * preempt-enabled, never under the FPU bracket or the ptl. There is no
 * cross-PMD FPU batching: kernel_fpu_begin/end save/restore is skipped for
 * kthreads (kswapd) and at most once per kernel entry otherwise, so
 * per-PMD bracketing is effectively free (see mm/lru_marie/simd_x86.c).
 */
struct marie_walk_arg {
	struct pglist_data	*pgdat;
	bool			force_scan;	/* bypass the coverage gate */
	unsigned long		*pfnbuf;	/* per-cpu deferred inc_tier worklist */
};

/*
 * Coverage gate: which of a task's PMDs are worth descending into?
 *
 * The walker is the ONLY thing that promotes a mapped folio to the head gen
 * ahead of reclaim, so whatever it declines to look at cannot be protected
 * from selection. Two candidate answers to "was anything in these 2 MiB
 * touched", in order of preference:
 *
 * 1. The hardware accessed bit on the PMD entry itself. x86 sets the A bit
 *    in EVERY paging-structure entry it walks, not just the leaf, so a
 *    non-leaf PMD's A bit means "some PTE under me was used to translate an
 *    address since it was last cleared". One lockless read answers for 512
 *    PTEs, and it is the CPU's own record rather than an inference: a 6 GiB
 *    task costs 3072 reads per pass instead of 1.5M. This is what MGLRU
 *    uses (should_clear_pmd_young / walk_pmd_range).
 *
 * 2. The rmap-fed bloom, when 1 is unavailable. Kept as the fallback, but
 *    note what it is: a list of PMDs lru_marie_look_around() has flagged,
 *    and look_around only runs from folio_referenced(), i.e. only once
 *    reclaim is ALREADY walking a folio it isolated out of the oldest gen.
 *    So a region enters the bloom only after it has first been reclaimed
 *    from unprotected -- the walker cannot see freshly-faulted memory at
 *    all, and the filter is primed by the very pages it failed to save.
 *    Measured on 6.12.74 with the bloom as the sole gate: a 6 GiB anon set
 *    rewritten in full every 33 ms lost 107k pages (418 MB) to swap in 30 s
 *    with MemFree never below 20 GiB, 22k of them refaulting straight back.
 *
 * Whoever consumes the bit must also clear it -- the CPU only ever sets it
 * -- which is what turns "used at some point" into "used since the last
 * pass". Clearing needs the PMD lock; a failed trylock just leaves the bit
 * set, costing one redundant scan next pass and never a missed access.
 * Clear BEFORE scanning the PTEs, so an access that races the scan re-sets
 * the bit and is caught next pass rather than being dropped between the two.
 *
 * mm_has_notifiers(): a secondary MMU (KVM) can touch a page without the
 * CPU walking these page tables, so here a clear A bit is a false negative.
 * Excluded from the fast path, and treated as "assume young" on the leaf
 * path -- the same two exclusions MGLRU makes.
 */
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
static bool marie_nonleaf_young_usable(struct mm_struct *mm)
{
	return arch_has_hw_nonleaf_pmd_young() && !mm_has_notifiers(mm);
}

static void marie_pmd_clear_young(struct vm_area_struct *vma, pmd_t *pmd,
				  unsigned long addr)
{
	spinlock_t *ptl = pmd_lockptr(vma->vm_mm, pmd);

	if (!spin_trylock(ptl))
		return;
	/* Re-validate under the lock: it must still be a page-table PMD. */
	if (pmd_present(*pmd) && !pmd_leaf(*pmd))
		pmdp_test_and_clear_young(vma, addr, pmd);
	spin_unlock(ptl);
}

/*
 * marie_walk_pmd_leaf - age a PMD-mapped THP.
 *
 * Marie registers only .pmd_entry and no .pte_entry, so pagewalk hands
 * huge PMDs straight to the walker and never splits them -- and
 * pte_offset_map_lock() then fails on them, which is how PMD-mapped THPs
 * came to be skipped entirely. With transparent_hugepage=always that is
 * most of a task's anon: 3.8 GiB of the 6 GiB probe above. A huge PMD has
 * no PTEs to scan; the A bit in the entry IS the folio's accessed bit, so
 * consume it here and promote the folio directly.
 */
static void marie_walk_pmd_leaf(struct marie_walk_arg *arg,
				struct vm_area_struct *vma, pmd_t *pmd,
				unsigned long addr, pmd_t val)
{
	struct pglist_data *pgdat = arg->pgdat;
	unsigned long pfn = pmd_pfn(val);
	bool notifiers = mm_has_notifiers(vma->vm_mm);
	spinlock_t *ptl;
	bool young = false;
	u8 s;

	if (marie_pmd_devmap(val))
		return;
	if (!pmd_young(val) && !notifiers)
		return;
	if (!pfn_valid(pfn))
		return;
	/* This pass is per pgdat. */
	if (pfn < pgdat->node_start_pfn || pfn >= pgdat_end_pfn(pgdat))
		return;
	if (!marie_state || pfn >= marie_state_size)
		return;

	ptl = pmd_lock(vma->vm_mm, pmd);
	/* pmd_same: the entry may have been collapsed/split since the
	 * lockless read above, in which case @pfn no longer describes it. */
	if (pmd_same(*pmd, val) && pmd_trans_huge(*pmd))
		young = !!pmdp_test_and_clear_young(vma, addr, pmd);
	spin_unlock(ptl);

	if (!young && !notifiers)
		return;

	s = READ_ONCE(marie_state[pfn]);
	if (!(s & MARIE_PFN_TRACKED))
		return;
	marie_state_inc_tier_seeded(pfn, s);
}
#else
static bool marie_nonleaf_young_usable(struct mm_struct *mm)
{
	return false;
}

static void marie_pmd_clear_young(struct vm_area_struct *vma, pmd_t *pmd,
				  unsigned long addr)
{
}

static void marie_walk_pmd_leaf(struct marie_walk_arg *arg,
				struct vm_area_struct *vma, pmd_t *pmd,
				unsigned long addr, pmd_t val)
{
}
#endif /* CONFIG_TRANSPARENT_HUGEPAGE */

static int marie_walk_pmd_range(pmd_t *pmd, unsigned long start,
			      unsigned long end, struct mm_walk *walk)
{
	struct marie_walk_arg *arg = walk->private;
	struct vm_area_struct *vma = walk->vma;
	pte_t *pte_table, *orig_pte;
	spinlock_t *ptl;
	unsigned long pmd_addr;
	pmd_t val;
	unsigned long bitmap[MARIE_SIMD_PTE_BITMAP_LONGS] = { 0 };
	unsigned long *pfnbuf = arg->pfnbuf;
	unsigned int bit, next_bit, nr = 0, i;

	if (!vma)
		return 0;

	pmd_addr = start & PMD_MASK;
	val = pmdp_get_lockless(pmd);

	if (!pmd_present(val) || is_huge_zero_pmd(val))
		return 0;

	/*
	 * PMD-mapped THP: no PTEs below to scan, and the entry's own A bit
	 * is the folio's accessed bit. Handled and consumed there.
	 */
	if (pmd_leaf(val)) {
		marie_walk_pmd_leaf(arg, vma, pmd, pmd_addr, val);
		/*
		 * Same reschedule point the PTE path takes at its tail. With
		 * transparent_hugepage=always most of a task's PMDs leave
		 * through here, so skipping it would let a pass run
		 * arbitrarily long without yielding.
		 */
		cond_resched();
		return 0;
	}

	/*
	 * Coverage gate for the 512 PTEs below -- see the block above
	 * struct marie_walk_arg for why the hardware bit is preferred over
	 * the bloom and what the bloom's blind spot costs. force_scan
	 * bypasses both.
	 */
	if (!arg->force_scan) {
		if (marie_nonleaf_young_usable(walk->mm)) {
			if (!pmd_young(val))
				return 0;	/* nothing in these 2 MiB */
			marie_pmd_clear_young(vma, pmd, pmd_addr);
		} else if (!marie_bloom_test(arg->pgdat->node_id, pmd_addr)) {
			return 0;
		}
	}

	/*
	 * pte_offset_map_lock returns pte_base + pte_index(start), which may
	 * not be at the start of the page table.  The SIMD kernel must receive
	 * the page-table base (index 0) so that:
	 *   (a) the 512-entry scan does not walk past the end of the page, and
	 *   (b) bit N in the output bitmap corresponds to pte_base[N], making
	 *       "pte_base + bit" the correct per-entry pointer in the loop.
	 *
	 * Keep orig_pte (= pte_base + pte_index(start)) for pte_unmap_unlock.
	 */
	pte_table = pte_offset_map_lock(walk->mm, pmd, start, &ptl);
	if (!pte_table)
		return 0;
	orig_pte = pte_table;

	/*
	 * Phase 1: scan the whole PMD under one self-contained FPU bracket
	 * (single-shot begin + SIMD scan + end). The bracket wraps only the
	 * scan, so it closes before the young-bit loop below and never spans
	 * PMDs or bloom-miss stretches. On simd=0 the begin/end are no-ops
	 * and the scalar scan runs directly.
	 */
	lru_marie_simd_young_pte_mask(pte_table - pte_index(start), bitmap);

	for (bit = find_first_bit(bitmap, 512); bit < 512; bit = next_bit) {
		unsigned long addr = pmd_addr + bit * PAGE_SIZE;
		pte_t *pte = orig_pte - pte_index(start) + bit;
		pte_t ptent;
		unsigned long pfn, fpfn;
		struct folio *folio;
		u8 s;

		/* Next set bit, computed once: it drives both the lookahead
		 * prefetch below and the loop advance (the for-increment), so
		 * we pay one find_next_bit per young bit instead of the two
		 * for_each_set_bit would cost (its internal scan plus ours). */
		next_bit = find_next_bit(bitmap, 512, bit + 1);

		/* Prefetch the hot line for the next young folio. The
		 * scattered access on this path is the per-PFN state byte
		 * marie_state[pfn]: it gates the TRACKED check and is then
		 * RMW'd by the tier bump, so prefetch it write-intent. struct
		 * page (page->flags) is touched only by the cross-node
		 * folio_pgdat() check below, elided on single-node systems --
		 * prefetch it just there. Sparse bitmap iteration defeats the
		 * HW prefetcher, so the explicit lookahead hides the L2/L3
		 * miss. */
		if (next_bit < 512) {
			pte_t next_ptent = ptep_get(orig_pte - pte_index(start)
						    + next_bit);
			unsigned long next_pfn = pte_pfn(next_ptent);

			if (pte_present(next_ptent) && pfn_valid(next_pfn)) {
				if (next_pfn < marie_state_size)
					prefetchw(&marie_state[next_pfn]);
				if (nr_online_nodes > 1)
					__builtin_prefetch(pfn_to_page(next_pfn), 0, 3);
			}
		}

		/* Only process PTEs within the [start, end) walk range. */
		if (addr < start || addr >= end)
			continue;

		ptent = ptep_get(pte);
		if (!pte_present(ptent) || pte_special(ptent))
			continue;

		pfn = pte_pfn(ptent);
		if (is_zero_pfn(pfn) || !pfn_valid(pfn))
			continue;

		folio = pfn_folio(pfn);

		/* Skip cross-node folios -- this pass is per pgdat. The
		 * folio_pgdat() read of page->flags is the only struct-page
		 * access on this path, so gate it on nr_online_nodes: on a
		 * single-node system every folio is on arg->pgdat, so the
		 * check (and the struct-page prefetch above) is pure overhead.
		 * On UMA builds nr_online_nodes is the constant 1, so the
		 * compiler drops this entirely. */
		if (nr_online_nodes > 1 && folio_pgdat(folio) != arg->pgdat)
			continue;

		/* Read the per-PFN state byte once. It gates the lock-free
		 * TRACKED pre-filter here (only TRACKED folios carry a live
		 * Marie tier; folio->lru is no longer a Marie-state signal),
		 * and on a hit it seeds the tier bump below so that skips a
		 * second load of the same now-cache-hot byte. folio_pfn() is
		 * pointer arithmetic; compute it once too. */
		fpfn = folio_pfn(folio);
		if (!marie_state || fpfn >= marie_state_size)
			continue;
		s = READ_ONCE(marie_state[fpfn]);
		if (!(s & MARIE_PFN_TRACKED))
			continue;

		if (!ptep_test_and_clear_young(vma, addr, pte))
			continue;

		/*
		 * Phase 2 collect: this folio is young + TRACKED and we have
		 * just cleared its accessed bit under the ptl. Defer the tier
		 * bump -- record the PFN and drain it after the ptl is dropped
		 * (Phase 3 below), so the heavy marie_state_inc_tier (the
		 * saturate -> in-place promote) never runs under the ptl or the
		 * FPU bracket. nr <= 512 = the bitmap's bit capacity, so pfnbuf
		 * (sized to it) never overflows.
		 */
		pfnbuf[nr++] = fpfn;
	}

	pte_unmap_unlock(orig_pte, ptl);

	/*
	 * Phase 3: drain the collected young PFNs into marie_state_inc_tier
	 * outside the ptl and outside any FPU bracket -- lock-free and
	 * preempt-enabled. This is the dominant per-PMD cost (the saturate
	 * -> in-place promote); keeping it off the preempt-disabled path is
	 * the whole point of the collect/drain split. inc_tier re-reads the
	 * per-PFN byte and re-checks TRACKED, so a rare PFN recycle between
	 * the clear above and here is self-correcting (at worst one stray
	 * bump). Prefetch the next byte write-intent -- inc_tier RMWs it.
	 */
	for (i = 0; i < nr; i++) {
		if (i + 1 < nr)
			prefetchw(&marie_state[pfnbuf[i + 1]]);
		marie_state_inc_tier(pfnbuf[i]);
	}
	cond_resched();
	return 0;
}

static const struct mm_walk_ops marie_walk_ops = {
	.pmd_entry	= marie_walk_pmd_range,
	.walk_lock	= PGWALK_RDLOCK,
};

static void marie_walk_one_mm(struct mm_struct *mm, struct marie_walk_arg *arg)
{
	if (!mmap_read_trylock(mm))
		return;
	walk_page_range(mm, 0, TASK_SIZE, &marie_walk_ops, arg);
	mmap_read_unlock(mm);
}

/**
 * marie_walk_pgdat - run one walker pass for @pgdat.
 *
 * Atomically claims the per-pgdat deadline; concurrent reclaimers /
 * kswapd cycles either advance the deadline or no-op. The walker
 * snapshots running mm_struct's via for_each_process under RCU,
 * walks each via the SIMD pmd_entry handler (which bumps the global
 * per-PFN tier state in place), then advances the global aging epoch
 * and rotates the bloom. There are no per-mlv queues to drain.
 *
 * Safe from any context that allows brief sleeping (cond_resched in
 * the inner walk).
 */
void marie_walk_pgdat(struct pglist_data *pgdat)
{
	int nid = pgdat->node_id;
	unsigned long deadline;
	struct marie_walk_ctx *ctx;
	struct task_struct *p;
	struct marie_walk_arg arg = {
		.pgdat = pgdat,
		/*
		 * force_scan disabled: the cold-bloom force_scan was the
		 * dominant kswapd startup latency under fault-burst (full
		 * PMD scan = ~45 ms on memhog 2.5 GB). The walker's role is
		 * purely tier promotion; folio_check_references' Marie gate
		 * (vmscan.c) is independent of walker state because
		 * lru_marie_mark_accessed funnels external access into tier rather
		 * than PG_referenced, so reclaim functions correctly even
		 * with an unprimed bloom. Bloom is warmed lazily by
		 * lru_marie_look_around during the first reclaim cycle's rmap walk.
		 */
		.force_scan = false,
	};
	int i;

	if (nid >= MAX_NUMNODES)
		return;	/* defensive */

	/* Atomic test-and-claim deadline for this pgdat. */
	deadline = (unsigned long)atomic_long_read(&marie_walker_next[nid]);
	if (time_before(jiffies, deadline))
		return;
	if ((unsigned long)atomic_long_cmpxchg(&marie_walker_next[nid],
					       (long)deadline,
					       (long)(jiffies + marie_walker_interval(pgdat))) != deadline)
		return;	/* lost race to another reclaimer */

	/*
	 * Pin to this CPU and reentrancy-claim its per-CPU walker ctx.
	 * The walker iterates up to MARIE_WALK_MAX_MMS mm_struct's per pass
	 * and walks each up to TASK_SIZE; running the entire pass with
	 * preempt_disable() makes cond_resched() inside marie_walk_pmd_range
	 * a no-op and starves the rest of the system to RCU stall under
	 * sustained memory pressure (observed as desktop stutter then
	 * freeze on real hardware). migrate_disable() keeps us on the
	 * CPU whose marie_walker_ctx we own, while marie_walker_busy stops a
	 * preempted-and-resumed reclaimer from reaching marie_walk_pgdat
	 * for a different pgdat on the same CPU and clobbering the
	 * in-flight snapshot.
	 */
	migrate_disable();
	if (this_cpu_cmpxchg(marie_walker_busy, 0, 1) != 0) {
		migrate_enable();
		return;
	}

	ctx = this_cpu_ptr(&marie_walker_ctx);
	ctx->n_mms = 0;
	arg.pfnbuf = ctx->pfnbuf;	/* per-cpu deferred inc_tier worklist */

	rcu_read_lock();
	for_each_process(p) {
		struct mm_struct *mm = READ_ONCE(p->mm);

		if (!mm || ctx->n_mms >= MARIE_WALK_MAX_MMS)
			continue;
		if (!mmget_not_zero(mm))
			continue;
		ctx->mms[ctx->n_mms++] = mm;
	}
	rcu_read_unlock();

	/*
	 * Walk preemptibly.  Each PMD opens and closes its own FPU bracket
	 * around just the SIMD scan (marie_walk_pmd_range), so no bracket is
	 * ever held across PMDs or across the mm walk, and the heavy tier
	 * bumps run after the ptl is dropped -- the preempt-disabled window
	 * is bounded to a single PMD's scan + young-bit clear, with a
	 * cond_resched between PMDs.
	 */
	for (i = 0; i < ctx->n_mms; i++) {
		marie_walk_one_mm(ctx->mms[i], &arg);
		/*
		 * mmput_async, not mmput: if our mmget_not_zero above pinned the
		 * last reference (the owning task exited mid-walk), a plain mmput
		 * here drops to zero and enters __mmput -> exit_mmap, which takes
		 * mm->mmap_lock. marie_walk_pgdat runs from kswapd's balance_pgdat
		 * with fs_reclaim held; taking mmap_lock under fs_reclaim closes
		 * the cycle against the execve path that takes mmap_lock then
		 * allocates (fs_reclaim) via mas_alloc_nodes. Caught by lockdep
		 * as a circular dependency and reproduced as a desktop hang under
		 * memory pressure with concurrent fork/exec. MGLRU solves the
		 * same problem the same way in iterate_mm_list (mm/vmscan.c).
		 */
		mmput_async(ctx->mms[i]);
	}

	/*
	 * No pass-end aging clock any more. This used to bump a global
	 * marie_aging_epoch[type] here, which reclaim compared against a
	 * per-gen recycle stamp to decide whether it could skip
	 * folio_check_references(). That comparison was removed (see
	 * marie_state_shrink_lruvec): the bump was unconditional -- it fired
	 * even when the pass walked no mms, and for FILE it could not observe
	 * its own subject at all, since unmapped page cache has no PTEs for a
	 * page-table walker to find -- so what it published was elapsed time
	 * dressed as coverage. Walker tier promotion is materialised inline by
	 * marie_state_inc_tier during the per-PMD walk, so nothing else
	 * accumulates to flush here.
	 */

	/*
	 * Pass-end bloom rotation: the inactive filter has accumulated
	 * Producer (look_around) feedback during this reclaim window;
	 * promote it to active so the next pass scans those PMDs. The
	 * old active is recycled as the new inactive, cleared of stale
	 * bits.
	 */
	marie_bloom_swap(nid);

	/* Release the per-CPU ctx claim before allowing migration. */
	this_cpu_write(marie_walker_busy, 0);
	migrate_enable();
}

/**
 * lru_marie_age_node - kswapd's pre-reclaim aging hook.
 *
 * MGLRU's `lru_gen_age_node()` analogue. Called from kswapd_age_node()
 * before direct reclaim machinery runs, so the gen ring has fresh
 * hot/cold ordering by the time pressure builds. Delegates to the
 * per-pgdat walker; rate-limiting is internal so calling on every
 * kswapd cycle is fine.
 */
void lru_marie_age_node(struct pglist_data *pgdat, struct scan_control *sc)
{
	marie_walk_pgdat(pgdat);
}
EXPORT_SYMBOL_GPL(lru_marie_age_node);

/**
 * marie_walker_init - one-shot init for the walker.
 *
 * Initialises per-pgdat bloom-filter spinlocks. Bitmaps themselves
 * are lazily allocated by marie_bloom_set() on first Producer hit
 * (under PTL, GFP_ATOMIC). Called from marie_init() in mm/lru_marie/core.c.
 */
void marie_walker_init(void)
{
	int nid;

	for (nid = 0; nid < MAX_NUMNODES; nid++)
		spin_lock_init(&marie_blooms[nid].lock);
}
