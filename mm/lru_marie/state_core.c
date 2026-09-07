// SPDX-License-Identifier: GPL-2.0
/*
 * Marie per-PFN state array — allocation, init, global counters, and the
 * gen-ring bookkeeping/mutation primitives (head advance, growth
 * threshold, move_to_gen, tier promotion, teardown).
 *
 * Implements the public storage declared in state.h: the flat
 * marie_state[] array indexed by PFN, the cycling head-gen counter,
 * and the per-(gen, type) install counters that drive aging. All of
 * these are allocated once at subsys_initcall time and never freed
 * for the lifetime of the kernel.
 *
 * Sizing rule: the array covers PFNs [0, max_pfn). max_pfn is bounded
 * by MARIE_MAX_SUPPORTED_PFN (the 32-bit PFN gate latched in
 * marie_init), so worst-case footprint is 4 GiB. Realistic configs
 * are 4-64 MiB. NUMA holes and reserved regions read as zero
 * (untracked) and incur only sequential-read cost during scans.
 *
 * This file holds the PFN-state algebra layer: it knows about bytes,
 * bits, and gens, but nothing about scan_control, lruvec, or folio
 * reclaim policy (see state_reclaim.c) or the folio-facing hook
 * surface (see state_folio.c).
 */

#define pr_fmt(fmt) "marie_state: " fmt

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/bitops.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/log2.h>
#include <linux/jump_label.h>
#include <linux/spinlock.h>
#include <linux/list.h>
#include <linux/lru_marie.h>
#include <linux/memblock.h>
#include <linux/memcontrol.h>
#include <linux/minmax.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/mm_inline.h>
#include <linux/mmzone.h>
#include <linux/oom.h>
#include <linux/printk.h>
#include <linux/sched.h>		/* cond_resched */
#include <linux/sched/signal.h>
#include <linux/swap.h>
#include <linux/vmalloc.h>
#include <linux/vm_event_item.h>
#include <linux/vmstat.h>
#include <linux/percpu.h>

#include "../internal.h"	/* struct scan_control, shrink_folio_list */
#include "state_compat.h"	/* MARIE_FOLIO_FLAGS, marie_shrink_folio_list, marie_account_reclaim */
#include "account.h"
#include "pfn_install.h"
#include "state.h"

u8 *marie_state;
unsigned long marie_state_size;

/*
 * Latches true once marie_state[] is allocated (first enable) and never
 * flips back -- the array lives for the kernel's lifetime. Gates the
 * page-free hook so stale TRACKED bits are wiped at the buddy handoff
 * even across a Marie disable transition (when lru_marie_enabled() is
 * already false but the drain walk is still in flight). See
 * marie_state_ready() in <linux/lru_marie.h>.
 */
DEFINE_STATIC_KEY_FALSE(marie_state_ready_key);
EXPORT_SYMBOL_GPL(marie_state_ready_key);

/*
 * Per-(gen, type) live folio population, maintained node-wide by the
 * marie_gen_occ_inc/dec abstraction. Desktop/global-only: a single global
 * per-(gen, type) counter drives both the oldest-gen scan target
 * (marie_find_oldest_occupied_mlv) and the "is any anon tracked?" signal read
 * by marie_file_floor_protect.
 */
atomic_long_t marie_gen_occupied[MARIE_PFN_NR_GENS][2];

/*
 * Count of the above that are non-zero, per type. See state.h for why it is
 * maintained at the choke point instead of being derived on demand.
 */
atomic_t marie_nr_occupied_gens[2];

/*
 * GLOBAL aging clock. Desktop/global-only Marie has no per-memcg reclaim, so a
 * single ring serves the whole node (replacing the retired per-mlv head/clock):
 *   marie_head_gen      youngest/install gen (atomic; cmpxchg on advance)
 *   marie_gen_installs  install-cadence counter (pages onto the head gen)
 * All shared across lruvecs (installs run under different per-lruvec lru_locks),
 * hence atomic; the u32s are racy heuristics (READ_ONCE/WRITE_ONCE is enough).
 */
atomic_t marie_head_gen[ANON_AND_FILE];
atomic_long_t marie_gen_installs[ANON_AND_FILE];


struct marie_bitmap marie_track_bm[2][MARIE_PFN_NR_ZONES_ENCODED][MARIE_PFN_NR_GENS];

/*
 * Allocate the per-PFN state array. Called from marie_init() after
 * the 32-bit PFN gate is latched, so max_pfn is guaranteed to fit
 * in the supported range.
 *
 * kvmalloc lets the array fall back to vmalloc on systems where a
 * physically contiguous allocation is unavailable; the array is
 * accessed strictly by PFN index and does not require contiguity.
 * GFP_KERNEL is safe here — initcall context can sleep.
 */
int __init marie_state_init(void)
{
	unsigned long bytes;
	int g, ty, z;

	bytes = max_pfn * sizeof(u8);
	if (!bytes) {
		pr_err("max_pfn is zero; refusing to initialise\n");
		return -EINVAL;
	}

	marie_state = kvmalloc(bytes, GFP_KERNEL | __GFP_ZERO);
	if (!marie_state) {
		pr_err("failed to allocate %lu-byte per-PFN state array\n",
		       bytes);
		return -ENOMEM;
	}
	marie_state_size = max_pfn;

	/*
	 * L2 sizing (marie_l2_shift/marie_l2_nbits from max_pfn and
	 * marie_max_l2_pages_per_bit) + the shared range-lock array. Must run
	 * before any marie_bm_init() call (those need marie_l2_nbits).
	 */
	if (marie_bm_global_init())
		goto state_oom;

	/* Per-(type, zone, gen) L1/L2 bitmaps: 64 total. */
	for (ty = 0; ty < 2; ty++) {
		for (z = 0; z < MARIE_PFN_NR_ZONES_ENCODED; z++) {
			for (g = 0; g < MARIE_PFN_NR_GENS; g++) {
				if (marie_bm_init(&marie_track_bm[ty][z][g]))
					goto bm_oom;
			}
		}
	}

	/*
	 * Latch the page-free hook on now that marie_state[] exists. Never
	 * disabled -- the array is never freed, and TRACKED bits can persist
	 * into a disable transition, so the hook must keep wiping them.
	 */
	static_branch_enable(&marie_state_ready_key);

	/*
	 * Seed the install-cadence inputs before the first install so the cadence
	 * never compares against a 0 base (which would advance the head on every
	 * install). At init each type owns nothing, so base is just free memory
	 * and gens is 0 -- the slowest-filling end of the schedule.
	 */
	marie_recompute_growth_base(0);
	marie_recompute_growth_base(1);

	pr_info("allocated state %lu B + %u tracking bitmaps (max_pfn=%lu, l2_shift=%u)\n",
		bytes, 2 * MARIE_PFN_NR_ZONES_ENCODED * MARIE_PFN_NR_GENS,
		max_pfn, marie_l2_shift);
	return 0;

bm_oom:
	for (ty = 0; ty < 2; ty++)
		for (z = 0; z < MARIE_PFN_NR_ZONES_ENCODED; z++)
			for (g = 0; g < MARIE_PFN_NR_GENS; g++)
				marie_bm_free(&marie_track_bm[ty][z][g]);
	kvfree(marie_bm_range_locks);
	marie_bm_range_locks = NULL;
state_oom:
	kvfree(marie_state);
	marie_state = NULL;
	return -ENOMEM;
}

/*
 * Global generation frame primitives (desktop/global-only). A single aging
 * clock per type (marie_head_gen[type]) serves the whole node: the per-PFN
 * byte stores a gen VALUE, and its age is that value read against the global
 * head_gen. The _mlv suffix is historical (the per-lruvec frame is gone).
 */

/*
 * Oldest gen (walking out from the global head) where the node has live folios.
 */
int marie_find_oldest_occupied_mlv(int type)
{
	int head = atomic_read(&marie_head_gen[type]);
	int i;

	for (i = 1; i < MARIE_PFN_NR_GENS; i++) {
		int slot = (head + i) & (MARIE_PFN_NR_GENS - 1);

		if (atomic_long_read(&marie_gen_occupied[slot][type]) > 0)
			return slot;
	}
	return -1;
}

/*
 * Install-cadence threshold inputs. The threshold itself is derived per install
 * by marie_gen_growth_threshold() (state.h) from two parts:
 *
 *   thr = max(base[type] >> (NR_GENS + 1 - gens), floor)
 *
 *   base[type] = this type's own pages + free memory: what it could grow into.
 *              Not memtotal -- memory the OTHER type is holding is not headroom
 *              this one can have without evicting it, so sizing against memtotal
 *              overstates the room and understates how finely to stratify.
 *   gens       = how many of this type's generations currently hold anything,
 *              read LIVE (marie_nr_occupied_gens, maintained at the gen-occupancy
 *              choke point). This is the whole point of the shape: the shift, and
 *              so the cadence, is a function of how full the ring already is.
 *   floor      = base >> (ilog2(NR_GENS) + gen_floor_shift): the cadence never
 *              runs faster than one generation per 1/NR_GENS of the same base,
 *              i.e. the ring is never asked to age faster than it would take to
 *              span what this type could hold. Measured against the same
 *              quantity as the schedule deliberately: a floor stated against
 *              "memory this type does NOT hold" instead grows when the other
 *              type grows, so a type squeezed by the other one would be pinned
 *              to a single generation -- 2 GiB of anon beside 28 GiB of file
 *              wants a 3.75 GiB cadence it can never reach.
 *
 *              Since the schedule is base >> (NR_GENS + 1 - gens), floor and
 *              schedule meet at gens == ilog2(NR_GENS) - gen_floor_shift, so the
 *              knob says exactly one thing: the generation count from which the
 *              schedule is allowed to act. At the 0 default it acts only as a
 *              brake near saturation, which is the arm that measured clean; at
 *              its 6 cap the floor drops below the schedule's own minimum and
 *              the schedule is unfloored, which is the arm that regressed
 *              (head jammed 3 of 3 runs against 0 of 3, 11% more swapped).
 *              Swept 0..6 in QEMU against a bounded 3.2 GiB anon set on 4 GiB
 *              swept slowly enough that nothing is reclaimed at all, which laps
 *              the ring on installs alone -- the same shape as the live jam,
 *              where anon reclaim stood at 1.2% of the set. The default is the
 *              only setting that keeps a spare slot:
 *
 *                        gens  head    largest gen
 *                 base     8   jammed      24%
 *                 k=0      7   free        15%
 *                 k=1      8   jammed      29%
 *                 k=2..6   8   jammed    35..52%
 *
 *              Replicated, the baseline jammed 4 of 4 and k=0 none of 3, with
 *              largest-generation ranges disjoint (24-36% against 15-17%).
 *              Under a reclaim-heavy workload the two are instead
 *              indistinguishable, which is consistent: there the drain keeps up
 *              and the head never laps, so the brake has nothing to do.
 *
 * Why a function of gens at all. The head can only advance into an EMPTY slot
 * (marie_try_advance_head_mlv), so from an empty ring it gets exactly one lap;
 * if reclaim has not drained the slot it started from by the time it comes back
 * around, it stops there permanently and every subsequent install piles into
 * that one generation. Nothing in the previous formula -- max(occ/NR_GENS,
 * (memtotal-occ)/NR_GENS, memtotal/256) -- referenced ring occupancy, so there
 * was no signal that this had happened and no way to respond: an open loop whose
 * terminal state is a jammed head. Measured on a live 30 GiB desktop, both rings
 * reached all NR_GENS occupied and blocked, anon holding 56% of its set in the
 * head generation and file's head going 7% -> 41% in four minutes.
 *
 * Making the shift depend on gens closes the loop: few generations shift far and
 * advance quickly (fill the ring), many generations shift little and advance
 * slowly (give reclaim time to drain the oldest). Equilibrium sits where
 * S = N * base >> (NR_GENS + 1 - N), i.e. 2^(NR_GENS+1-N) = N * (1 + free/S),
 * which lands at N ~ 4.7..6.3 -- always short of NR_GENS, so a spare slot
 * survives and the head keeps moving. Generation sizes come out geometric while
 * the ring grows (base/512 up to base/4), which puts the finest resolution at the
 * oldest end, where reclaim actually chooses.
 *
 * It agrees with the formula it replaces exactly where that one was tuned: at
 * free -> 0 and N == 6 this is base/8 == occ/NR_GENS. It only relaxes where
 * there is free memory to relax into.
 *
 * The clean_min_ratio reserve term is gone rather than carried over. At
 * equilibrium with file sitting on its reserve it reproduces itself
 * (base >> 3 == reserve/NR_GENS), and coupling a protection policy to an aging
 * mechanism meant writing the protection knob silently changed how finely the
 * ring stratified.
 *
 * base is refreshed off the install path -- at each head advance and
 * at reclaim entry, so free memory is tracked when free memory is under pressure
 * -- while gens is read live. That split is deliberate: gens is the term that
 * has to be current for the loop to be a loop, and it is the cheap one (a single
 * atomic_read, versus NR_GENS reads of the most heavily written cachelines Marie
 * has).
 */
unsigned long marie_gen_growth_base[2];

void marie_recompute_growth_base(int type)
{
	unsigned long own;

	if (type == 0)
		own = global_node_page_state(NR_ACTIVE_ANON) +
		      global_node_page_state(NR_INACTIVE_ANON);
	else
		own = global_node_page_state(NR_ACTIVE_FILE) +
		      global_node_page_state(NR_INACTIVE_FILE);

	WRITE_ONCE(marie_gen_growth_base[type],
		   own + global_zone_page_state(NR_FREE_PAGES));
}

/*
 * Advance the global head if its next slot is empty. The gate
 * (gen_occupied[next]==0) guarantees nothing live sits at the slot the head
 * recycles, so it never aliases old folios.
 *
 * Returns true iff it advanced (the install-cadence caller resets its install
 * counter only on a real advance, so a blocked attempt -- next slot still
 * draining -- retries on the following install).
 *
 * Driven solely by install cadence under the installing lruvec's lru_lock (see
 * marie_folio_install). The former reclaim-time "occupied < 2" trigger was
 * removed: under concurrent global reclaim it fired on every shrink entry and
 * raced the head around the ring (~10^6 advances/run vs ~10^1 aging ticks),
 * destroying age stratification so the oldest gen held mixed-age (incl. hot)
 * folios -> rotation -> ~50% reclaim efficiency and OOM with swap free.
 */
bool marie_try_advance_head_mlv(int type)
{
	u8 head = (u8)atomic_read(&marie_head_gen[type]);
	u8 next = (head + 1) & (MARIE_PFN_NR_GENS - 1);

	if (atomic_long_read(&marie_gen_occupied[next][type]) != 0)
		return false;
	if (atomic_cmpxchg(&marie_head_gen[type], head, next) != head)
		return false;
	/*
	 * Real advance: resample this type's cadence base. Runs for
	 * install-cadence AND demand-pull advances. gens is not sampled here --
	 * the install path reads it live.
	 */
	marie_recompute_growth_base(type);
	return true;
}

/*
 * marie_gen_occ_inc/dec -- the SINGLE gen-occupancy interface -- now live as
 * static inline in state.h so pfn_install.h's install-side publisher
 * (marie_pfn_publish_inherit) routes through the same hook instead of bumping
 * marie_gen_occupied directly. See the contract above their definition there.
 */


/*
 * marie_state_drop_pfn - zero out every per-PFN tracking artifact for one
 * folio (state byte, (type, zone, gen) L1 bit, and the global
 * per-(gen, type) occupancy counter).
 *
 * Called from:
 *   marie_evict_locked      -- normal evict path
 *   marie_drain_pfn_locked  -- enable=0 sysfs flip; folio gets
 *                              returned to legacy LRU, the per-PFN
 *                              artifacts MUST be wiped or they
 *                              survive across the disabled window
 *                              as ghosts that wedge counters on
 *                              re-enable.
 *
 * No-op when the state byte is not TRACKED (defensive against
 * double-drop). Reads the (type, zone, gen) tuple from the byte BEFORE
 * zeroing it.
 *
 * marie_bm_clear is called unconditionally (harmless no-op if the bit is
 * already clear -- e.g. marie_evict_counters_only already retired it at
 * isolate-claim time); gen_occupied is decremented ONLY when
 * marie_bm_clear reports the real 1->0 transition, so a folio that is
 * simultaneously ISOLATED and reaching this path some other way (e.g. an
 * external del racing an in-flight reclaim) cannot double-debit. This is
 * the same discipline the isolate scan's own orphaned-bit cleanup uses.
 *
 * Counter settlement is NOT done here -- it belongs to the caller, which
 * knows its own locking context (see account.h's LOCKED vs ISOLATE split).
 * Both wrappers below hand the returned transition to the matching
 * marie_acct_settle_*(), so the debit is derived from the byte transition
 * rather than stated independently of it.
 */

/*
 * marie_state_untrack - atomically clear @pfn's state byte and retire its
 * scan bit. Returns the byte it replaced.
 *
 * The xchg is what makes the teardown exactly-once. The previous
 * read-then-write pair let two concurrent teardown paths both observe
 * TRACKED and both consider themselves the one that owed the debit; with
 * an xchg only one can observe the TRACKED byte, so only one can compute a
 * nonzero accounting delta from it.
 */
static u8 marie_state_untrack(unsigned long pfn)
{
	u8 s = xchg(&marie_state[pfn], 0);
	u8 g, zone, type_bit;

	if (!(s & MARIE_PFN_TRACKED))
		return s;

	marie_gen_occ_settle(pfn, s, 0);

	/* Scan index, derived from the byte; return value is advisory. */
	g = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	zone = (s & MARIE_PFN_ZONE_MASK) >> MARIE_PFN_ZONE_SHIFT;
	type_bit = (s & MARIE_PFN_TYPE_MASK) ? 1 : 0;
	marie_bm_retire(pfn, type_bit, zone, g);
	return s;
}

/* LOCKED context: caller holds folio's lru_lock with IRQs off. */
void marie_state_drop_pfn(struct folio *folio)
{
	unsigned long pfn;
	u8 s;

	if (!marie_state || !folio)
		return;

	pfn = folio_pfn(folio);
	if (pfn >= marie_state_size)
		return;

	s = marie_state_untrack(pfn);
	marie_acct_settle_locked(folio_lruvec(folio), folio_nr_pages(folio),
				 s, 0);
}
EXPORT_SYMBOL_GPL(marie_state_drop_pfn);

/*
 * ISOLATE context: caller holds no lock, IRQs on. Used by the survivor
 * putback's "folio became free" branch, which must undo the publish it
 * just made -- including the state byte, or lru_marie_uncharge_backstop()
 * would treat the dying folio as one that escaped reclaim entirely.
 */
void marie_state_drop_pfn_isolate(struct folio *folio)
{
	unsigned long pfn;
	u8 s;

	if (!marie_state || !folio)
		return;

	pfn = folio_pfn(folio);
	if (pfn >= marie_state_size)
		return;

	s = marie_state_untrack(pfn);
	marie_acct_settle_isolate(folio_lruvec(folio), folio_nr_pages(folio),
				  s, 0);
}

/*
 * marie_state_drop_pfn_at_free - canonical buddy-handoff cleanup.
 *
 * Invoked from mm/page_alloc.c::free_pages_prepare for every page about
 * to enter the buddy allocator. Eliminates the deferred-cleanup race
 * between marie_evict_counters_only (counters -1, TRACKED preserved) and
 * the next allocation at the same PFN: the moment the page is destined
 * for buddy, we wipe Marie's per-PFN bookkeeping so a subsequent
 * install_local starts from a clean state byte.
 *
 * Counters are NOT touched here -- they were either already balanced
 * by marie_evict_locked (the normal Marie del path) or pre-decremented
 * by marie_evict_counters_only (the reclaim isolate path), and the
 * page-free hook runs once per page regardless of which del path was
 * taken upstream.
 *
 * memcg_bitmap is intentionally untouched. folio_memcg is unsafe to
 * dereference at free time (the page is mid-uncharge); the stale bit
 * is harmless because the next install at this PFN under a different
 * memcg will re-set the new memcg's bitmap bit, and a memcg teardown
 * will free the bitmap wholesale.
 *
 * Lock-free: byte write, bitmap atomic-bit-clear, atomic_long_dec --
 * safe from any context including IRQ.
 */
void marie_state_drop_pfn_at_free(unsigned long pfn)
{
	u8 s, g, zone, type_bit;

	if (!marie_state || pfn >= marie_state_size)
		return;

	/*
	 * Cheap advisory filter only -- skip the flag scrub below for the
	 * overwhelming majority of PFNs Marie never tracked. The authoritative,
	 * exactly-once transition is the xchg further down; this read is
	 * deliberately not the one the accounting is derived from.
	 */
	s = marie_state[pfn];
	if (!(s & MARIE_PFN_TRACKED))
		return;

	/*
	 * A TRACKED folio reaching the buddy free path still carrying PG_lru
	 * bypassed Marie's evict (which clears both TRACKED and PG_lru under
	 * the folio_test_clear_lru claim). Leaving PG_lru set trips the
	 * "Bad page state |lru|" PAGE_FLAGS_CHECK_AT_FREE oops. Clear it
	 * here as the canonical last-resort: the folio is being freed
	 * (refcount 0) and Marie folios keep folio->lru as a self-loop
	 * (never linked onto a real lruvec list), so dropping PG_lru cannot
	 * corrupt any list. This is a mitigation for a residual reclaim
	 * accounting race (a Marie folio reaching free with TRACKED still
	 * set); marie_nr_folios IS settled below, so the only residue such a
	 * folio can now leave is its lru_size page weight, which is
	 * unresolvable here (memcg_data is already zeroed).
	 */
	{
		struct folio *f = page_folio(pfn_to_page(pfn));

		/*
		 * Invariant: a TRACKED folio must never reach the buddy free
		 * path still carrying PG_lru. Marie's evict clears both under
		 * the folio_test_clear_lru claim, and folio_batch_move_lru no
		 * longer re-stamps PG_lru onto a tracked folio (the mm/swap.c
		 * fix). VM_WARN_ON_ONCE flags a regression of that invariant in
		 * DEBUG_VM builds; it compiles to nothing in production, so the
		 * folio_test_lru below costs only a predicted-not-taken branch
		 * on an already-hot folio->flags. The trailing clear is the
		 * production last resort -- it degrades any future regression
		 * to a counter blip instead of a PAGE_FLAGS_CHECK_AT_FREE oops.
		 * Marie folios keep folio->lru detached from real lruvec lists,
		 * so clearing PG_lru here cannot corrupt a list.
		 */
		if (unlikely(folio_test_lru(f))) {
			VM_WARN_ON_ONCE_FOLIO(1, f);
			folio_clear_lru(f);
		}
		/*
		 * shrink_folio_list can re-set PG_active on a folio whose
		 * PG_lru is clear (Marie isolated it). PG_active is in
		 * PAGE_FLAGS_CHECK_AT_FREE; if still set here it would
		 * trigger bad_page in free_pages_prepare. Clear it
		 * unconditionally as a last-resort safety net.
		 */
		if (unlikely(folio_test_active(f)))
			folio_clear_active(f);

#ifdef CONFIG_LRU_GEN
		/*
		 * Scrub MGLRU gen/refs residue. LRU_GEN_MASK is in
		 * PAGE_FLAGS_CHECK_AT_FREE, so a leftover gen counter trips
		 * "Bad page state" in free_pages_prepare. With Marie masking
		 * lru_gen_enabled() off (see lru_gen_enabled()), no MGLRU
		 * writer stamps these onto a tracked folio, so this is the
		 * structural last resort that keeps any future regression a
		 * counter blip rather than a buddy-path oops -- independent of
		 * whether every lru_gen_enabled() reader stays correctly gated.
		 *
		 * PG_workingset is deliberately NOT cleared: Marie's eviction
		 * relies on the legacy workingset_eviction shadow encoding,
		 * which reads PG_workingset, and the bit is not in
		 * PAGE_FLAGS_CHECK_AT_FREE.
		 */
		if (unlikely(MARIE_FOLIO_FLAGS(f) & (LRU_GEN_MASK | LRU_REFS_MASK))) {
			VM_WARN_ON_ONCE_FOLIO(1, f);
			set_mask_bits(&MARIE_FOLIO_FLAGS(f), LRU_GEN_MASK | LRU_REFS_MASK, 0);
		}
#endif
	}

	/*
	 * xchg, not a plain store: the replaced byte is what the accounting
	 * delta is derived from, so it must be observed exactly once.
	 *
	 * Normally pred is already 0 by the time a folio gets here (isolate
	 * claimed it, evict wiped it, or lru_marie_uncharge_backstop settled
	 * it) and marie_acct_settle_count computes 0. The exception is a folio
	 * that was never charged to a memcg, hence never uncharged, hence
	 * never seen by the backstop: it arrives still holding its credit, and
	 * this is the last point anything runs for it. Settling the folio
	 * count here keeps marie_nr_folios exact for that corner instead of
	 * leaking +1 per occurrence. Its lru_size page weight is the one
	 * residue Marie cannot resolve, because memcg_data is already zeroed
	 * and folio_memcg is unsafe to dereference at this point -- see
	 * marie_acct_settle_count.
	 */
	s = xchg(&marie_state[pfn], 0);
	if (!(s & MARIE_PFN_TRACKED))
		return;
	marie_acct_settle_count(s, 0);
	marie_gen_occ_settle(pfn, s, 0);

	g = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	zone = (s & MARIE_PFN_ZONE_MASK) >> MARIE_PFN_ZONE_SHIFT;
	type_bit = (s & MARIE_PFN_TYPE_MASK) ? 1 : 0;

	/*
	 * Retire the scan-index bit. Occupancy was already settled from the
	 * byte transition above; marie_bm_retire only reconciles the index,
	 * and re-publishes the bit if the byte turns out to name this plane
	 * again (a concurrent re-install at this PFN).
	 */
	marie_bm_retire(pfn, type_bit, zone, g);
}

/*
 * lru_marie_uncharge_backstop - debit an escaped tracked folio at the last
 *                               point its owning memcg is still live.
 *
 * Called from mm/memcontrol.c::uncharge_folio, immediately before
 * folio->memcg_data is zeroed. This is the universal confluence point we
 * unify the counter debit on: every charged LRU folio is uncharged before
 * it reaches the buddy allocator (per the allocator's contract), and the
 * memcg is still readable here -- unlike the page-free hook, which runs
 * after the uncharge and therefore has no memcg to debit.
 *
 * This no longer DECIDES whether a debit is owed -- it just performs the
 * "accounting settled" state transition and lets the delta follow. Setting
 * ISOLATED takes pred (TRACKED && !ISOLATED) from 1 to 0 for a folio that
 * still held its credit, so marie_acct_settle_isolate derives exactly one
 * debit; for a folio already settled upstream (isolate claimed it, or the
 * survivor putback's freed branch wiped the byte) pred is already 0 and the
 * derived delta is 0. The old form read the two bits and decided for
 * itself, which is what let the putback publish -- a legitimate lifecycle
 * action that clears ISOLATED -- silently re-arm a second debit on a folio
 * this function had no business touching. See account.h.
 *
 * TRACKED is deliberately LEFT SET: marie_state_drop_pfn_at_free() still
 * needs it as its gate to scrub PG_lru / PG_active / LRU_GEN residue before
 * the buddy handoff, and it settles nothing further because pred is now 0.
 *
 * A cmpxchg loop rather than a plain OR: uncharge normally runs only once a
 * folio's refcount is genuinely dropping to zero, but the transition is the
 * thing the accounting is derived from, so it has to be exactly-once by
 * construction rather than by that argument holding.
 *
 * The debit bucket is reconstructed PURELY from the per-PFN state byte
 * (marie_acct_lru / marie_acct_zone) -- never from folio_lru_list(), which
 * the reclaim shrinker may have re-stamped with PG_active. No new per-PFN
 * storage is needed: the byte says which bucket, the still-live memcg says
 * whose counter.
 *
 * IRQ-tolerant: uncharge can run from softirq (folio_put on bio completion);
 * marie_acct_settle_isolate owns the local_irq_save/restore.
 */
void lru_marie_uncharge_backstop(struct folio *folio, struct mem_cgroup *memcg)
{
	unsigned long pfn;
	u8 s, g, type_bit, zone;

	if (!marie_state_ready() || !marie_state)
		return;
	pfn = folio_pfn(folio);
	if (pfn >= marie_state_size)
		return;

	s = READ_ONCE(marie_state[pfn]);
	do {
		if (!(s & MARIE_PFN_TRACKED) || (s & MARIE_PFN_ISOLATED))
			return;
	} while (!try_cmpxchg(&marie_state[pfn], &s,
			      (u8)(s | MARIE_PFN_ISOLATED)));

	/* @s is now the pre-transition byte we won. */
	marie_gen_occ_settle(pfn, s, (u8)(s | MARIE_PFN_ISOLATED));
	marie_acct_settle_isolate(mem_cgroup_lruvec(memcg, folio_pgdat(folio)),
				  folio_nr_pages(folio), s,
				  (u8)(s | MARIE_PFN_ISOLATED));

	/* Scan index, derived from the byte; return value is advisory. */
	g = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	zone = (s & MARIE_PFN_ZONE_MASK) >> MARIE_PFN_ZONE_SHIFT;
	type_bit = (s & MARIE_PFN_TYPE_MASK) ? 1 : 0;
	marie_bm_retire(pfn, type_bit, zone, g);
}

/*
 * marie_state_move_to_gen - relocate a tracked PFN to @target_gen, with a
 * matched marie_track_bm / gen_occupied update.
 *
 * ISOLATED gate: if marie_state[pfn] has ISOLATED set, this PFN is
 * exclusively owned by an in-flight isolate/putback right now -- decline
 * without touching anything. Because GEN and ISOLATED now live in the
 * SAME byte (unlike the prior marie_state/marie_age split), this check
 * and the relocation CAS are the same atomic operation: cur is re-read
 * fresh at the top of every retry iteration, and the cmpxchg below can
 * only succeed against the exact @cur it was checked against. If isolate
 * concurrently sets ISOLATED via its own CAS (see marie_evict_counters_
 * only) between this function's read and its cmpxchg, the cmpxchg simply
 * fails (the byte changed underneath it) and the retry re-reads, sees
 * ISOLATED now set, and declines cleanly. This closes the residual race
 * the old two-byte (marie_state + marie_age) split could not: there is
 * no gap left for a mutator's read-then-cmpxchg to straddle isolate's
 * ownership claim.
 *
 * Skipped if the folio is no longer tracked, is currently ISOLATED, or
 * already encodes @target_gen.
 *
 * Called from:
 *   marie_state_inc_tier promote-on-access (target_gen=head)
 *   marie_folio_demote (MADV_COLD)
 *   lru_marie_lazyfree (MADV_FREE)
 *   Marie defrag's post-migration restamp (defrag.c)
 * Putback (shrink_lruvec residue release) does NOT call this -- it has
 * its own accounting path since it is also the one clearing ISOLATED;
 * see the putback code below.
 */
void marie_state_move_to_gen(unsigned long pfn, u8 target_gen)
{
	u8 cur, new_byte, type, zone, old_gen;

	if (pfn >= marie_state_size)
		return;
	target_gen &= MARIE_PFN_NR_GENS - 1;

	/*
	 * Lock-free. The byte CAS is the entire commit: occupancy is derived
	 * from it (marie_gen_occ_settle), and the bitmap writes below are only
	 * index maintenance.
	 *
	 * This used to hold @pfn's L2 range claim across the whole sequence,
	 * to keep the byte and the scan bitmap -- two independently mutated
	 * authorities -- in agreement, because gen_occupied was maintained
	 * from the BITMAP's transitions. Under that scheme any disagreement
	 * corrupted a counter, so the two had to be updated atomically
	 * together. Now that gen_occupied is a projection of the byte
	 * (state.h's marie_gen_occ_settle), the bitmap is a lossy index and
	 * disagreement is harmless in both directions:
	 *
	 *   - a bit set where the byte no longer points: the scanner reads the
	 *     byte, mismatches, and self-heals the bit away. It moves no
	 *     counter, and the folio is still indexed at wherever the byte
	 *     does point.
	 *   - a bit not yet set where the byte already points: the scanner is
	 *     driven by SET bits, so it simply does not visit this PFN this
	 *     pass. Nothing observes the gap, and nothing can clear a bit that
	 *     is not set -- which is why the "stranded folio" the lock was
	 *     added to prevent cannot occur: publishing the new bit before
	 *     retiring the old one means there is no instant at which the
	 *     folio is absent from every plane.
	 *
	 * MARIE_PFN_ISOLATED still excludes isolate/putback, from a different
	 * angle (folio ownership, not index bookkeeping): it lives in the same
	 * byte as GEN, so the check below and the CAS are one atomic
	 * operation. If isolate lands its own ISOLATED CAS in between, ours
	 * fails, the retry re-reads, and we decline cleanly.
	 */
retry:
	cur = READ_ONCE(marie_state[pfn]);
	if (!(cur & MARIE_PFN_TRACKED) || (cur & MARIE_PFN_ISOLATED))
		return;

	old_gen = (cur & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	if (old_gen == target_gen)
		return;

	new_byte = (u8)((cur & ~MARIE_PFN_GEN_MASK) |
			((u8)target_gen << MARIE_PFN_GEN_SHIFT));

	if (cmpxchg(&marie_state[pfn], cur, new_byte) != cur)
		goto retry;

	/* We committed the transition, so we own its occupancy delta. */
	marie_gen_occ_settle(pfn, cur, new_byte);

	type = (cur & MARIE_PFN_TYPE_MASK) ? 1 : 0;
	zone = (cur & MARIE_PFN_ZONE_MASK) >> MARIE_PFN_ZONE_SHIFT;

	/*
	 * Index maintenance: publish the new plane BEFORE retiring the old one,
	 * so the folio is never missing from both at once.
	 *
	 * The retire must be marie_bm_retire, not a bare marie_bm_clear. Two
	 * concurrent move_to_gen calls on this PFN serialise their byte CASes
	 * but not their index writes, so A(G->H) and B(H->G) can interleave as
	 * A.set(H), B.set(G), A.clear(G), B.clear(H) -- both publishes wiped,
	 * byte naming a plane with no bit anywhere. marie_bm_retire's
	 * post-clear re-read restores it. See its comment in state.h.
	 */
	marie_bm_set(&marie_track_bm[type][zone][target_gen], pfn);
	marie_bm_retire(pfn, type, zone, old_gen);
}
EXPORT_SYMBOL_GPL(marie_state_move_to_gen);

/*
 * marie_state_inc_tier - promote-to-head-on-access.
 *
 * Runs from folio_mark_accessed() WITHOUT lru_lock, racing the lock-free
 * reclaim isolate and the lru_lock-held install publish; the relocation
 * itself is delegated to marie_state_move_to_gen()'s cmpxchg retry loop.
 * Name kept from the prior (saturating-tier) design for API stability --
 * see state.h's byte-layout block for the retirement of the intermediate
 * tier step: any access now moves a tracked, non-isolated folio straight
 * to the head gen.
 */
/*
 * __marie_state_inc_tier - promote-on-access from an already-read state
 * byte @cur. The caller guarantees pfn < marie_state_size.
 *
 * @cur's TRACKED/ISOLATED bits are checked from this snapshot as a cheap
 * early-out; the authoritative check happens inside
 * marie_state_move_to_gen() itself (it re-reads marie_state[pfn] fresh
 * before committing), so a stale @cur here only risks one wasted
 * atomic_read of marie_head_gen, never an incorrect commit.
 */
static void __marie_state_inc_tier(unsigned long pfn, u8 cur)
{
	u8 type, gen, head;

	if (!(cur & MARIE_PFN_TRACKED) || (cur & MARIE_PFN_ISOLATED))
		return;
	type = (cur & MARIE_PFN_TYPE_MASK) ? 1 : 0;
	gen = (cur & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;

	head = (u8)atomic_read(&marie_head_gen[type]);
	if (head == gen)
		return;
	marie_state_move_to_gen(pfn, head);
}

void marie_state_inc_tier(unsigned long pfn)
{
	if (pfn >= marie_state_size)
		return;
	__marie_state_inc_tier(pfn, READ_ONCE(marie_state[pfn]));
}
EXPORT_SYMBOL_GPL(marie_state_inc_tier);

/*
 * marie_state_inc_tier_seeded - promote-on-access when the caller already
 * holds the state byte and has bounds-checked @pfn (the walker, which
 * gated on the TRACKED bit just before clearing the young bit). Skips the
 * redundant reload + bound check; the cmpxchg loop still self-corrects
 * @cur.
 */
void marie_state_inc_tier_seeded(unsigned long pfn, u8 cur)
{
	__marie_state_inc_tier(pfn, cur);
}
EXPORT_SYMBOL_GPL(marie_state_inc_tier_seeded);
