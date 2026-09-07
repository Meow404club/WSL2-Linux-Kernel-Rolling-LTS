// SPDX-License-Identifier: GPL-2.0
/*
 * mm/lru_marie/defrag.c -- Marie defragmentation, Step 1.
 *
 * Maintains a per-pageblock (gen, type) occupancy histogram off Marie's
 * single gen-occupancy choke-point (marie_gen_occ_inc/dec, state.h). The
 * histogram is the sufficient-statistic input to the cost-scored block
 * selector; this step is observability ONLY -- no migration, no reclaim,
 * no policy change.
 *
 * Correctness is checkable by construction: because the only writers are the
 * choke-point hooks, summing the histogram over all blocks must reproduce the
 * global marie_gen_occupied counter for every (gen, type). The read-only
 * sysfs node verifies this. (gen_occupied and its per-block mirror are bumped
 * by two separate atomics, so a sum taken under active churn can differ by the
 * number of in-flight transitions; the invariant is exact at quiescence --
 * read it after the load settles.)
 */
#define pr_fmt(fmt) "lru_marie_defrag: " fmt

#include <linux/atomic.h>
#include <linux/build_bug.h>
#include <linux/kobject.h>
#include <linux/migrate.h>		/* migrate_pages, alloc_migration_target */
#include <linux/migrate_mode.h>		/* MIGRATE_SYNC_LIGHT */
#include <linux/mm.h>			/* max_pfn */
#include <linux/mmzone.h>
#include <linux/numa.h>			/* NUMA_NO_NODE */
#include <linux/pageblock-flags.h>
#include <linux/printk.h>
#include <linux/sched.h>		/* cond_resched */
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/swap.h>		/* lru_add_drain */
#include <linux/sysfs.h>
#include <linux/vmalloc.h>
#include <linux/vmstat.h>		/* node_page_state */
#include <linux/lru_marie.h>		/* lru_marie_enabled */

#include "../internal.h"		/* folio_isolate_lru, migration_target_control */
#include "defrag_compat.h"		/* marie_defrag_prep_allocated (6.14 refcount split) */
#include "defrag.h"
#include "state.h"			/* marie_gen_occupied, MARIE_PFN_NR_GENS */

struct marie_defrag_block_hist *marie_defrag_hist;
unsigned long marie_defrag_nr_blocks;

/*
 * Blocks that evacuation cannot free, so selection must not keep picking them.
 *
 * The cost ranking itself is right and is not what changed here. Every source
 * block yields the same thing -- one 2 MiB pageblock -- so ranking by the work
 * to empty it means the cheapest candidate is the best candidate, and the ideal
 * source is a block with almost nothing in it. Evacuating one folio to free a
 * pageblock beats evacuating five hundred to free the same pageblock.
 *
 * What the ranking lacks is a feasibility precondition. Cost is summed over
 * Marie-TRACKED folios only, so it is silent about the other 511 pages, and a
 * block Marie barely owns is either almost empty -- the ideal source -- or
 * pinned by something Marie will never move: slab, page tables, reserved pages,
 * any kernel allocation. Those two look identical to the scorer, and the second
 * kind is unusable at any price.
 *
 * Worse, the ranking has no memory, so an unusable block stays the cheapest and
 * is picked again on the very next run, and again after that. Measured on a live
 * 30 GiB desktop: 198 kcompactd wakes produced 2 migrated folios and ONE freed
 * pageblock, while reclaim evicted 9.4 GiB of page cache to manufacture the
 * high-order blocks compaction was failing to produce.
 *
 * So the fix is a filter in front of the ranking, not a different ranking:
 * verify a candidate before committing to it (marie_defrag_block_freeable) and
 * remember the rejects here, one bit per pageblock.
 *
 * The map has to be forgotten eventually -- a block becomes freeable again the
 * moment whatever pinned it is freed, and nothing reports that. But a periodic
 * clear is the wrong shape: the interval has no relation to the event, so it is
 * simultaneously too slow (a block that freed up immediately waits out the
 * timer) and too fast (a page table that will never move is re-scanned forever),
 * and any interval chosen without measuring is just a number.
 *
 * Forget on DEMAND instead. The only moment stale knowledge costs anything is a
 * run that could not fill its evacuation budget from the candidates it was
 * allowed to consider -- that, and only that, is when it is worth paying to
 * re-examine the rejects. A run that filled its budget learned nothing by
 * forgetting, so it doesn't.
 */
static unsigned long *marie_defrag_skip;
/*
 * rej_pinned  -- rejected because something in the block will never move.
 * rej_nothing -- rejected because there was nothing in it worth moving (a
 *                pageblock-sized folio is already the contiguous block defrag
 *                exists to produce).
 * skipped     -- blocks collect() passed over because they are already marked.
 * amnesty     -- times the map was cleared as stale.
 *
 * The two rejection reasons are counted apart because they say different things
 * about the scorer. rej_nothing climbing means it keeps ranking blocks that are
 * ALREADY the goal state at the very front, which is a cost-model artifact:
 * cost counts folios, so one THP reads as occ == 1 and scores cheaper than any
 * file block holding more than five folios. rej_pinned climbing means the front
 * of the list is instead full of blocks Marie can never empty. Neither counts
 * candidates that were merely surplus to the evacuation budget -- that is
 * headroom working as intended, not an event.
 */
static atomic_long_t marie_defrag_tot_rej_pinned;
static atomic_long_t marie_defrag_tot_rej_nothing;
static atomic_long_t marie_defrag_tot_skipped;
static atomic_long_t marie_defrag_tot_amnesty;

/*
 * Where an evacuated block ended up. freed is already counted separately; these
 * are the two ways it can fail, and they were indistinguishable while only the
 * successes were counted.
 *
 * left_occ  -- Marie folios remain, so the block was not fully evacuated.
 *              Two ways in: isolation could not take a folio (counted
 *              separately as iso_fail, and measured at zero over ~32 million
 *              folios), or migration took it and handed it back. In practice
 *              this is the second one, and it tracked target supply: 56% while
 *              the harvest cap starved alloc_target, ~1% once the cap was
 *              derived from the requested order.
 * left_pin  -- no Marie folios remain and the block is still not free: it holds
 *              something Marie does not track. These are marked in the skip map
 *              (that is the outcome-learning path); counted here so the rate is
 *              visible rather than inferred from the map's size.
 *
 * evac is the denominator: blocks actually evacuated this run. freed + left_occ
 * + left_pin == evac, which is what makes the three readable as a breakdown.
 */
static atomic_long_t marie_defrag_tot_evac;
static atomic_long_t marie_defrag_tot_left_occ;
static atomic_long_t marie_defrag_tot_left_pin;
/*
 * Folios isolate_block wanted but could not take (try_get or isolate_lru lost
 * to a concurrent reclaimer). Kept even though it has never been non-zero:
 * Marie's reclaim isolate is lock-free, so the race is real, and this is the
 * only thing that would tell us it started happening.
 */
static atomic_long_t marie_defrag_tot_iso_fail;
/*
 * alloc_target could not produce a destination, by the ORDER that was asked
 * for. The pool only ever held orders up to the harvest gate, so "every failure
 * is at order > gate" and "the pool ran dry at every order" are completely
 * different diagnoses -- supply shape versus supply volume -- and `dst` alone
 * cannot tell them apart. Indexed by the source folio's order.
 */
static atomic_long_t marie_defrag_tot_alloc_fail[NR_PAGE_ORDERS];

/*
 * Destinations handed to migrate_pages, i.e. restamp entries recorded. This is
 * the denominator for `restamped`: without it the two could only be compared
 * against `migrated`, which counts PAGES while a restamp entry is one FOLIO,
 * and the ratio was only bounded, never known.
 */
static atomic_long_t marie_defrag_tot_dst;

/* Max source blocks EVACUATED per run. */
#define MARIE_DEFRAG_TOPK	16

/*
 * Max source blocks CONSIDERED per run. Deliberately much larger than the
 * evacuation budget: candidates are ranked by evacuation cost, and cost says
 * nothing about whether a block can be freed at all, so a run must be able to
 * fall past a run of unfreeable cheap blocks and still fill its budget. When
 * these two were the same constant, sixteen pinned blocks meant the run did
 * nothing -- measured on a live desktop as 198 kcompactd wakes producing one
 * freed pageblock.
 *
 * Held in a file-static array rather than on the stack: 128 candidates is 2 KiB,
 * well past the frame budget. Single-owner is already enforced by
 * marie_defrag_busy, which is what makes a shared array safe here.
 */
#define MARIE_DEFRAG_CANDK	128

/*
 * Age-neutrality scratch (see the design comment above marie_defrag_freectx).
 * Two pre-allocated (pfn, gen) tables plus the single-owner trylock guarding
 * them. Declared here because marie_defrag_scratch_alloc() below sizes them.
 */
#define MARIE_DEFRAG_GEN_NONE	0xff	/* srcmap miss -> fall back to oldest */

struct marie_defrag_pg {
	u32	pfn;
	u8	gen;
};

static struct marie_defrag_pg	*marie_defrag_srcmap;	/* src pfn -> gen (isolation) */
static struct marie_defrag_pg	*marie_defrag_restamp;	/* dst pfn + intended gen */
static unsigned long		 marie_defrag_scratch_cap;	/* entries per table */
static atomic_t			 marie_defrag_busy = ATOMIC_INIT(0);

/*
 * Allocate the per-pageblock histogram covering [0, max_pfn) rounded up to a
 * whole pageblock. ~1 MiB on a 30 GiB box (≈15k blocks x 64 B). kvmalloc:
 * accessed by block index only, so physical contiguity is not required. Lives
 * for the kernel's lifetime, like marie_state.
 */
static int __init marie_defrag_hist_alloc(void)
{
	unsigned long bytes, pb_pages = 1UL << pageblock_order;

	BUILD_BUG_ON(MARIE_PFN_NR_GENS != MARIE_DEFRAG_NGENS);

	marie_defrag_nr_blocks = (max_pfn + pb_pages - 1) >> pageblock_order;
	if (!marie_defrag_nr_blocks)
		return -EINVAL;

	bytes = marie_defrag_nr_blocks * sizeof(struct marie_defrag_block_hist);
	marie_defrag_hist = kvmalloc(bytes, GFP_KERNEL | __GFP_ZERO);
	if (!marie_defrag_hist)
		return -ENOMEM;

	/*
	 * Non-fatal: without the map, verification still rejects unfreeable
	 * blocks, it just cannot remember them between runs.
	 */
	marie_defrag_skip = kvmalloc(BITS_TO_LONGS(marie_defrag_nr_blocks) *
				     sizeof(unsigned long), GFP_KERNEL | __GFP_ZERO);

	pr_info("per-pageblock histogram: %lu blocks (order %u), %lu KiB\n",
		marie_defrag_nr_blocks, pageblock_order, bytes >> 10);
	return 0;
}

/*
 * Would evacuating this block's Marie folios actually free the pageblock?
 *
 * Marie only ever evacuates folios it tracks, so the block is a viable source
 * only if everything else in it is either free or something migration can move.
 * A single slab page, page table, PageReserved page or plain kernel allocation
 * pins it forever as far as this driver is concerned.
 *
 * That question is exactly what has_unmovable_pages() answers, so ask it rather
 * than reimplementing the classification -- it also gets the cases a hand-rolled
 * version gets wrong: hugetlb pages whose hstate does not support migration,
 * CMA blocks, HWPoison, PageOffline. It was made non-static for this caller.
 *
 * migratetype=MIGRATE_MOVABLE: only reached for blocks marie_defrag_collect()
 * already filtered to that type, and it is what makes a CMA block answer
 * "unmovable" here -- the conservative and correct answer for Marie, which has
 * no business evacuating one.
 *
 * flags=0: no MEMORY_OFFLINE. HWPoison and PageOffline pages then count as
 * blockers, which is right -- offlining may be able to walk away from them,
 * defrag cannot move them.
 *
 * Inexact, and taken without zone->lock; see the definition. A wrong answer
 * costs one wasted or one skipped candidate, never correctness.
 */
static bool marie_defrag_block_freeable(unsigned long blk)
{
	unsigned long start = blk << pageblock_order;
	unsigned long end = start + (1UL << pageblock_order);

	if (!pfn_valid(start))
		return false;
	if (end > max_pfn)
		end = max_pfn;
	if (start >= end)
		return false;

	return !marie_defrag_block_has_unmovable(start, end);
}

/*
 * Did the pageblock actually become free? Every page buddy-free, nothing else.
 *
 * Distinct from "Marie has no folios left here", which is all the per-block
 * histogram can say. A block can reach zero Marie occupancy and still hold
 * slab, page tables, or zsmalloc pages -- and zsmalloc in particular passes
 * has_unmovable_pages() as __PageMovable while Marie never touches it, because
 * Marie only evacuates folios carrying MARIE_PFN_TRACKED. Conflating the two
 * is why the freed counter used to overstate the high-order yield.
 *
 * Racy without zone->lock, deliberately: a false "not free" costs one skip-map
 * entry that demand-driven amnesty will lift.
 */
static bool marie_defrag_block_is_free(unsigned long blk)
{
	unsigned long start = blk << pageblock_order;
	unsigned long end = start + (1UL << pageblock_order);
	unsigned long pfn;

	if (!pfn_valid(start))
		return false;
	if (end > max_pfn)
		end = max_pfn;

	for (pfn = start; pfn < end; pfn++) {
		struct page *page = pfn_to_page(pfn);

		if (page_ref_count(page))
			return false;
		if (PageBuddy(page)) {
			unsigned int order = buddy_order_unsafe(page);

			if (order <= MAX_PAGE_ORDER)
				pfn += (1UL << order) - 1;
		}
	}
	return true;
}

/*
 * Non-atomic __test_bit/__set_bit: every access is under marie_defrag_busy,
 * which already enforces a single owner. The atomic forms would only buy a
 * bus lock nobody is contending for.
 */
/*
 * Is there anything here for evacuation to move?
 *
 * A pageblock whose tracked content is a single pageblock-sized folio is
 * ALREADY the 2 MiB contiguous block defrag exists to produce. It is the goal
 * state, not a source -- and isolate_block skips such a folio for exactly that
 * reason ("compound fills block: already contiguous").
 *
 * The cost scorer cannot see this, because the histogram counts FOLIOS: one THP
 * makes the block read occ == 1, which on the anon coefficient scores 16 --
 * cheaper than any file block holding more than five folios. So THP-only blocks
 * sort to the very front of the candidate list, get selected, contribute
 * nothing, and do it again on the next run. With THP=always they are not rare:
 * this desktop had 296 of them resident against a 128-entry candidate list, so
 * the list could be composed entirely of blocks that cannot yield anything.
 *
 * The outcome check does not catch it either. Nothing was isolated, so
 * occupancy stays non-zero, which that check reads as "migration failed this
 * time, try again later" -- true for a locked page, wrong for a folio that will
 * be skipped on every future run for a structural reason. Hence this predicate,
 * up front, rather than another special case in the outcome.
 *
 * Cheap: the same 512-entry scan the freeable check makes, and it stops at the
 * first folio worth moving.
 */
static bool marie_defrag_block_has_evacuable(unsigned long blk)
{
	unsigned long start = blk << pageblock_order;
	unsigned long end = start + (1UL << pageblock_order);
	unsigned long pfn;

	if (end > marie_state_size)
		end = marie_state_size;

	for (pfn = start; pfn < end; pfn++) {
		struct folio *folio;
		u8 st = READ_ONCE(marie_state[pfn]);

		if (!(st & MARIE_PFN_TRACKED) || !pfn_valid(pfn))
			continue;
		folio = pfn_folio(pfn);
		/* Mirrors isolate_block's skip, so the two agree on what counts. */
		if (folio_test_large(folio) &&
		    folio_nr_pages(folio) >= (1UL << pageblock_order))
			continue;
		return true;
	}
	return false;
}

static bool marie_defrag_skipped(unsigned long blk)
{
	return marie_defrag_skip && test_bit(blk, marie_defrag_skip);
}

static void marie_defrag_skip_mark(unsigned long blk)
{
	if (marie_defrag_skip)
		__set_bit(blk, marie_defrag_skip);
}

/*
 * Demand-driven amnesty -- see marie_defrag_skip. Called only by a run that
 * came up short, so the next run reconsiders everything.
 */
static void marie_defrag_skip_amnesty(void)
{
	if (!marie_defrag_skip)
		return;
	atomic_long_inc(&marie_defrag_tot_amnesty);
	bitmap_zero(marie_defrag_skip, marie_defrag_nr_blocks);
}

/*
 * Pre-allocate the two age-neutrality scratch tables (srcmap, restamp) once, at
 * boot, so the under-pressure defrag path never allocates. Sized to the worst
 * case run: MARIE_DEFRAG_TOPK fully-occupied source blocks. Non-fatal on failure
 * -- defrag then runs without age correction (plain head install), like a per-run
 * alloc that failed used to. ~64 KiB per table on an order-9 pageblock.
 */
static void __init marie_defrag_scratch_alloc(void)
{
	unsigned long cap = (unsigned long)MARIE_DEFRAG_TOPK << pageblock_order;
	size_t bytes = cap * sizeof(struct marie_defrag_pg);

	marie_defrag_srcmap = kvmalloc(bytes, GFP_KERNEL);
	marie_defrag_restamp = kvmalloc(bytes, GFP_KERNEL);
	if (!marie_defrag_srcmap || !marie_defrag_restamp) {
		kvfree(marie_defrag_srcmap);
		kvfree(marie_defrag_restamp);
		marie_defrag_srcmap = marie_defrag_restamp = NULL;
		marie_defrag_scratch_cap = 0;
		pr_warn("age-neutrality scratch alloc failed; defrag runs without age correction\n");
		return;
	}
	marie_defrag_scratch_cap = cap;
	pr_info("age-neutrality scratch: %lu entries x2 (%zu KiB)\n",
		cap, (bytes * 2) >> 10);
}

/*
 * Completeness invariant. For each (gen, type), the histogram summed over all
 * blocks must equal the global marie_gen_occupied -- otherwise a gen
 * transition escaped the choke-point. O(nr_blocks x NCLASS); debug-on-read.
 */
static int marie_defrag_invariant_report(char *buf, int len, int cap)
{
	long hist_sum[MARIE_DEFRAG_NGENS][MARIE_DEFRAG_NTYPES] = {};
	unsigned long b;
	int g, t, mism = 0;

	for (b = 0; b < marie_defrag_nr_blocks; b++) {
		struct marie_defrag_block_hist *h = &marie_defrag_hist[b];

		for (g = 0; g < MARIE_DEFRAG_NGENS; g++)
			for (t = 0; t < MARIE_DEFRAG_NTYPES; t++)
				hist_sum[g][t] +=
					atomic_read(&h->count[g * MARIE_DEFRAG_NTYPES + t]);
	}

	for (g = 0; g < MARIE_DEFRAG_NGENS; g++) {
		for (t = 0; t < MARIE_DEFRAG_NTYPES; t++) {
			long occ = atomic_long_read(&marie_gen_occupied[g][t]);

			if (hist_sum[g][t] != occ) {
				mism++;
				len += scnprintf(buf + len, cap - len,
					"MISMATCH gen %d type %d: hist %ld occ %ld\n",
					g, t, hist_sum[g][t], occ);
			}
		}
	}
	len += scnprintf(buf + len, cap - len, "invariant %s (%d mismatch)\n",
			 mism ? "FAIL" : "OK", mism);
	return len;
}

/* marie_defrag_invariant_report() is consumed by the defrag_stats read below. */

/*
 * ---------------------------------------------------------------------
 * Static cost scorer (Step 2) -- read-only block_cost ranking.
 * ---------------------------------------------------------------------
 *
 * block_cost(B) = sum over a block's occupants of a per-(age, type) cost
 * weight: a sufficient-statistic dot product over the histogram.
 * Coefficients are STATIC relative units encoding the evacuation ladder below
 * (Step 5 replaces them with boot-measured / learned ns). Read-only: this
 * ranks and reports candidate source blocks; it migrates nothing.
 */

/* Relative per-occupant evacuation cost (abstract units). */
#define MARIE_DEFRAG_C_DROP	 1u	/* cold, clean, unmapped file: ~unlink (near free) */
#define MARIE_DEFRAG_C_MOVE_FILE	 3u	/* warm clean unmapped file: copy + xarray, no TLB */
#define MARIE_DEFRAG_C_MOVE_ANON	16u	/* anon: always mapped -> copy + rmap + TLB shootdown */

/*
 * Per-occupant cost. Drop-eligibility is K=1: only the single gen
 * marie_find_oldest_occupied_mlv(FILE) reports as the oldest still-occupied
 * file gen is drop-eligible (near free); anon is TLB-bound regardless of age
 * because it is always mapped. @oldest_file is that gen (or -1 if no file gen
 * is occupied at all, e.g. immediately post-boot -- nothing is droppable
 * then). marie_find_oldest_occupied_mlv scans forward from head+1, so it can
 * never return head itself: the head's own (youngest) gen is structurally
 * excluded, not just numerically unlikely. Volatile mapped/dirty refinement
 * is left to phase 2 / the later learned coeffs; phase 1 scores on the stable
 * (gen, type) class.
 */
static inline u32 marie_defrag_coeff(int gen, int type, int oldest_file)
{
	if (type)	/* FILE */
		return (oldest_file >= 0 && gen == oldest_file) ? MARIE_DEFRAG_C_DROP
								 : MARIE_DEFRAG_C_MOVE_FILE;
	return MARIE_DEFRAG_C_MOVE_ANON;	/* ANON */
}

static long marie_defrag_block_occupancy(const struct marie_defrag_block_hist *h)
{
	long occ = 0;
	int c;

	for (c = 0; c < MARIE_DEFRAG_NCLASS; c++)
		occ += atomic_read(&h->count[c]);
	return occ;
}

static u64 marie_defrag_block_cost(const struct marie_defrag_block_hist *h, int oldest_file)
{
	u64 cost = 0;
	int g;

	for (g = 0; g < MARIE_DEFRAG_NGENS; g++) {
		long ca = atomic_read(&h->count[g * MARIE_DEFRAG_NTYPES + 0]);
		long cf = atomic_read(&h->count[g * MARIE_DEFRAG_NTYPES + 1]);

		cost += (u64)ca * marie_defrag_coeff(g, 0, oldest_file);
		cost += (u64)cf * marie_defrag_coeff(g, 1, oldest_file);
	}
	return cost;
}

struct marie_defrag_cand {
	u64		cost;
	unsigned long	blk;
	long		occ;
};

/* Insert (cost, blk, occ) into the ascending top-K array holding *n entries. */
static void marie_defrag_topk_insert(struct marie_defrag_cand *top, int *n, u64 cost,
			    unsigned long blk, long occ)
{
	int i;

	if (*n == MARIE_DEFRAG_CANDK && cost >= top[MARIE_DEFRAG_CANDK - 1].cost)
		return;
	i = (*n < MARIE_DEFRAG_CANDK) ? (*n)++ : MARIE_DEFRAG_CANDK - 1;
	for (; i > 0 && top[i - 1].cost > cost; i--)
		top[i] = top[i - 1];
	top[i].cost = cost;
	top[i].blk = blk;
	top[i].occ = occ;
}

/*
 * Phase 1: rank every non-empty MIGRATE_MOVABLE pageblock by block_cost into
 * the ascending top-K array. Shared by the read-only candidate listing and the
 * compaction driver. Returns the count collected (<= MARIE_DEFRAG_TOPK).
 */
static int marie_defrag_collect(struct marie_defrag_cand *top, unsigned long *scanned_out,
		       unsigned long *movable_out)
{
	int oldest_file = marie_find_oldest_occupied_mlv(1);
	unsigned long b, scanned = 0, movable = 0;
	int n = 0;

	for (b = 0; b < marie_defrag_nr_blocks; b++) {
		struct marie_defrag_block_hist *h = &marie_defrag_hist[b];
		unsigned long pfn = b << pageblock_order;
		long occ = marie_defrag_block_occupancy(h);

		if (occ == 0)
			continue;
		scanned++;
		if (!pfn_valid(pfn))
			continue;
		if (get_pageblock_migratetype(pfn_to_page(pfn)) != MIGRATE_MOVABLE)
			continue;
		movable++;
		/* Known unfreeable: do not let it crowd out a usable candidate. */
		if (marie_defrag_skipped(b)) {
			atomic_long_inc(&marie_defrag_tot_skipped);
			continue;
		}
		marie_defrag_topk_insert(top, &n, marie_defrag_block_cost(h, oldest_file),
				b, occ);
	}
	if (scanned_out)
		*scanned_out = scanned;
	if (movable_out)
		*movable_out = movable;
	return n;
}

/* marie_defrag_collect() is the selection used by marie_defrag_topn (scanned/
 * movable out-params unused now; pass NULL). */

/*
 * ---------------------------------------------------------------------
 * Migration front-end (Step 3 + Step 6 target selection).
 * ---------------------------------------------------------------------
 *
 * Evacuate the Marie-tracked (movable LRU) folios of the cheapest source
 * pageblocks via the core migrate_pages() machinery -- design P5: Marie defrag
 * contributes the SELECTION, not the migration. folio_isolate_lru() routes
 * through Marie's del (untracking the source folio, decrementing the histogram
 * at the source block); migrate_pages() copies + remaps and re-adds each
 * destination via folio_add_lru() -> lru_marie_add_folio() (re-installing it,
 * incrementing the histogram at the destination block). The move is therefore
 * histogram-coherent for free.
 *
 * Step 6 fixes step 3's net-yield problem: rather than letting the buddy place
 * targets (alloc_migration_target, which splits high-order free blocks to find
 * order-0 target pages and so fragments them), Marie defrag pre-harvests target pages
 * from the sub-pageblock free HOLES of OTHER partial movable blocks
 * (marie_defrag_harvest_block via __isolate_free_page, capped just below the
 * requested order so a free block big enough to be the answer is never broken)
 * and hands them out
 * (marie_defrag_alloc_target / marie_defrag_free_target, mirroring compaction_alloc/free).
 * Evacuees thus fill existing holes -- the source blocks free into order-9
 * blocks, the harvested-from blocks get denser, and no high-order block is
 * split. DROP of cold-dead clean file is still deferred to step 4; this step
 * MOVES every occupant.
 */

/*
 * Cumulative diagnostic counters, exposed ONLY via the /sys .../defrag_stats
 * read while the debug flag (marie_defrag_stats) is on -- for an A/B test vs
 * stock compaction (read totals before and after a window, diff). No per-run
 * dmesg output. Maintained unconditionally (a few atomics per run) so the A/B
 * window does not depend on when the flag was flipped.
 */
static unsigned int marie_defrag_stats;		/* debug flag: gates the stats read */
static atomic_long_t marie_defrag_tot_runs, marie_defrag_tot_move,
		     marie_defrag_tot_migrated, marie_defrag_tot_dropped,
		     marie_defrag_tot_freed, marie_defrag_tot_restamped;
/*
 * Master switch (sysfs /sys/kernel/mm/lru_marie/defrag): 1 = Marie defrag
 * REPLACES stock compaction on both kcompactd paths (DEFAULT -- the build
 * already opted in via CONFIG_LRU_MARIE_DEFRAG, so an installed defrag kernel
 * is active out of the box); 0 = stock kernel compaction, Marie dormant.
 */
static unsigned int marie_defrag_enabled = 1;
static atomic_long_t marie_defrag_fires;	/* proactive replacements done (stat) */

/*
 * Safety killswitch (sysfs /sys/kernel/mm/lru_marie/defrag_drop): 1 (DEFAULT)
 * = urgent/direct compaction may DROP cold-dead clean file as designed; 0 =
 * never DROP, even when the caller's may_drop says the path is urgent --
 * compaction falls back to MOVE-only, identical to the proactive path. Lets
 * an operator disable the drop-and-refault-risk rung alone if it proves
 * harmful on a given workload, without losing Marie defrag's MOVE-based
 * compaction (that would require the coarser .../defrag master switch).
 */
static unsigned int marie_defrag_drop_enabled = 1;

/*
 * The harvest must not consume what it is trying to produce, so free runs at or
 * above the requested order are left alone. That threshold is not a tunable:
 * stock compaction makes the identical decision in suitable_migration_target(),
 *
 *	int order = cc->order > 0 ? cc->order : pageblock_order;
 *	if (buddy_order_unsafe(page) >= order)
 *		return false;
 *
 * and marie_defrag_req carries the same order, so it is derived per run rather
 * than compiled in. Both kcompactd paths land on pageblock_order here: the
 * demand path is woken for order 9, and the proactive path passes 0, which
 * means "no specific requester" and falls back to pageblock_order exactly as
 * stock's cc->order == 0 case does.
 *
 * It used to be a fixed 3, five binary orders below what the stated reason
 * ("never break a near-pageblock free block") requires. A lone free order-8 run
 * is by definition one whose buddy is occupied -- two free order-8 buddies in a
 * pageblock would already have merged into the order-9 -- so harvesting it
 * cannot destroy an order-9 that would otherwise form. The cost of the extra
 * caution was severe: on a 30 GiB desktop 64%% of free memory sat in orders 4-8
 * and was refused, alloc_target could not serve a single folio above order 3,
 * and 53%% of everything isolate_block took had to be handed straight back
 * (16,008,960 folios isolated, 7,507,497 destinations, 1.003 pages each --
 * every successful migration was a single-page folio).
 *
 * Note what this gives up: the old cap incidentally protected nearly-empty
 * blocks, whose free space tends to be one large run, from being harvested --
 * and those are precisely the cheapest SOURCE candidates. Only occ == 0 blocks
 * are skipped explicitly (marie_defrag_harvest_refill). Whether a low-occupancy
 * skip is also needed is left to measurement rather than a guessed threshold:
 * if it is, freed_ok/evac falls.
 */
static unsigned int marie_defrag_harvest_gate(unsigned int req_order)
{
	unsigned int order = req_order ? req_order : pageblock_order;

	return order ? order - 1 : 0;
}

/*
 * Age neutrality across migration. The generic migrate path re-adds the
 * destination via folio_add_lru() -> marie_folio_install(), which installs at
 * the HEAD (youngest) gen -- rejuvenating a folio that defrag merely relocated.
 * To undo that, after migration we move each dst off head back to the SOURCE
 * folio's own gen, so a cold source stays cold and a warm source stays warm:
 * the age *distribution* survives the relocation, not just its coldest end.
 *
 * The source gen is absolute, but head advances while the (possibly long)
 * MIGRATE_SYNC runs (the run's own dst installs drive the install-cadence
 * clock). Because gens are a mod-N ring, a source gen the advancing head has
 * LAPPED past ("beyond tail" -- its slot no longer exists) is indistinguishable
 * by number from a still-valid old gen, and replaying it could land at/ahead of
 * head and pin the head-advance gate (gen_occupied[next] != 0) into an
 * OOM-livelock. So the restamp keeps the absolute source gen ONLY while it
 * still lies within the current occupied arc [oldest .. head] (age <= oldest's
 * age); once lapped it falls back to the current oldest -- frame-relative, so it
 * can never pin the gate. See marie_defrag_restamp_target().
 *
 * TIER is preserved from the dst's own byte (migration copies PG_active /
 * PG_workingset, which install reads); only the GEN is corrected. Capturing the
 * source gen needs a side channel: folio_isolate_lru clears the source byte, and
 * the src<->dst pairing is first visible in alloc_target -- so isolation records
 * (src pfn, gen) into @srcmap (sorted, bsearch'd by alloc_target) and alloc_target
 * records (dst pfn, that gen) into @restamp for the tail to apply. pfn fits in
 * u32: Marie requires max_pfn < 2^32 (MARIE_MAX_SUPPORTED_PFN).
 *
 * Both tables are PRE-ALLOCATED once at init (marie_defrag_scratch_alloc), never
 * on the defrag path: defrag runs under memory pressure, where a per-run kvmalloc
 * could fail and silently drop the age correction. They are shared, so a single
 * owner runs at a time (marie_defrag_busy trylock in lru_marie_defrag_pgdat); a
 * concurrent caller skips (best-effort).
 */

/* Target free-page pool, harvested from partial blocks (compaction-style). */
struct marie_defrag_freectx {
	struct list_head freepages[NR_PAGE_ORDERS];
	unsigned long	 nr;		/* order-0-equivalent pages available now */
	unsigned long	 scan_cursor;	/* next block to harvest holes from */
	unsigned long	 harvested;	/* cumulative pages harvested (stats) */
	unsigned int	 harvest_max_order;	/* see marie_defrag_harvest_gate */
	/*
	 * Age-neutrality scratch: borrowed pointers into the pre-allocated
	 * global tables (NULL = init alloc failed -> fall back to head install).
	 * @srcmap: (src pfn, gen) captured at isolation, sorted by pfn.
	 * @restamp: (dst pfn, intended gen) recorded at alloc_target.
	 * Both bounded by @cap (the worst-case run occupancy).
	 */
	struct marie_defrag_pg	*srcmap;
	unsigned int		 srcmap_n;
	struct marie_defrag_pg	*restamp;
	unsigned int		 restamp_n;
	unsigned int		 cap;
};

/* A refill grabs at least this many pages per scan, to amortise the walk. */
#define MARIE_DEFRAG_HARVEST_BATCH	64

static void marie_defrag_harvest_refill(struct marie_defrag_freectx *fc, unsigned long min_pages);

static void marie_defrag_freectx_init(struct marie_defrag_freectx *fc,
				      unsigned int req_order)
{
	int o;

	for (o = 0; o < NR_PAGE_ORDERS; o++)
		INIT_LIST_HEAD(&fc->freepages[o]);
	fc->nr = 0;
	fc->scan_cursor = 0;
	fc->harvested = 0;
	fc->harvest_max_order = marie_defrag_harvest_gate(req_order);
	/*
	 * Borrow the pre-allocated scratch (single-owner via marie_defrag_busy).
	 * If the init alloc failed both stay NULL and the run degrades to the
	 * plain head install -- no age correction, but no per-run alloc either.
	 */
	fc->srcmap = marie_defrag_srcmap;
	fc->restamp = marie_defrag_restamp;
	fc->cap = (marie_defrag_srcmap && marie_defrag_restamp) ?
			(unsigned int)marie_defrag_scratch_cap : 0;
	fc->srcmap_n = fc->restamp_n = 0;
}

/* Sort key for @srcmap so alloc_target can binary-search a migrating src's gen. */
static int marie_defrag_pg_cmp(const void *a, const void *b)
{
	u32 pa = ((const struct marie_defrag_pg *)a)->pfn;
	u32 pb = ((const struct marie_defrag_pg *)b)->pfn;

	return (pa > pb) - (pa < pb);
}

/*
 * Look up the gen captured at isolation for source @pfn. @srcmap is sorted by
 * pfn (marie_defrag_topn sorts it before migration starts). Returns the gen, or
 * MARIE_DEFRAG_GEN_NONE if not found (the caller then falls back to oldest).
 */
static u8 marie_defrag_srcmap_gen(struct marie_defrag_freectx *fc, unsigned long pfn)
{
	int lo = 0, hi = (int)fc->srcmap_n - 1;
	u32 key = (u32)pfn;

	while (lo <= hi) {
		int mid = (lo + hi) >> 1;
		u32 mp = fc->srcmap[mid].pfn;

		if (mp == key)
			return fc->srcmap[mid].gen;
		if (mp < key)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return MARIE_DEFRAG_GEN_NONE;
}

/*
 * Where to restamp a migrated dst, given its captured source gen and the CURRENT
 * ring (@head, @oldest for the dst's type). Keep the absolute source gen while
 * it still lies within the occupied arc [oldest .. head] -- i.e. its head-relative
 * age has not passed the oldest's age. Once the advancing head has lapped it
 * ("beyond tail", or an unknown source), promote to @oldest: frame-relative, so
 * it tracks head and can never re-populate the head-advance gate slot. Worst case
 * is thus either a benign install at head (a genuinely young source) or at oldest
 * (== the old always-oldest behaviour) -- never a gate pin.
 */
static int marie_defrag_restamp_target(u8 src_gen, u8 head, int oldest)
{
	unsigned int mask = MARIE_PFN_NR_GENS - 1;
	unsigned int age_src, age_oldest;

	if (src_gen == MARIE_DEFRAG_GEN_NONE)
		return oldest;
	src_gen &= mask;
	age_src = (head - src_gen) & mask;
	age_oldest = (head - (u8)oldest) & mask;
	return age_src <= age_oldest ? (int)src_gen : oldest;
}

/*
 * get_new_folio for migrate_pages: hand out a target from the harvested pool,
 * split down to the source folio's order. Mirrors compaction_alloc(): when the
 * pool runs dry it refills by scanning FORWARD for more partial blocks to
 * harvest holes from (marie_defrag_harvest_refill), so a drained pool selects new
 * destinations and continues instead of failing the rest of the batch. NULL is
 * returned only when no holes remain anywhere -- migrate_pages then puts the
 * source folio back.
 */
static struct folio *marie_defrag_alloc_target(struct folio *src, unsigned long data)
{
	struct marie_defrag_freectx *fc = (struct marie_defrag_freectx *)data;
	int order = folio_order(src);
	struct page *freepage;
	unsigned long size;
	int so;

	for (so = order; so < NR_PAGE_ORDERS; so++)
		if (!list_empty(&fc->freepages[so]))
			break;
	if (so == NR_PAGE_ORDERS) {
		marie_defrag_harvest_refill(fc, max_t(unsigned long, 1UL << order,
					     MARIE_DEFRAG_HARVEST_BATCH));
		for (so = order; so < NR_PAGE_ORDERS; so++)
			if (!list_empty(&fc->freepages[so]))
				break;
		if (so == NR_PAGE_ORDERS) {
			if (order < NR_PAGE_ORDERS)
				atomic_long_inc(&marie_defrag_tot_alloc_fail[order]);
			return NULL;
		}
	}

	freepage = list_first_entry(&fc->freepages[so], struct page, lru);
	size = 1UL << so;
	list_del(&freepage->lru);
	while (so > order) {
		so--;
		size >>= 1;
		list_add(&freepage[size].lru, &fc->freepages[so]);
		set_page_private(&freepage[size], so);
	}
	fc->nr -= 1UL << order;

	/*
	 * Turn the harvested raw free page into a refcounted allocation. Since
	 * 6.14 post_alloc_hook() no longer sets the refcount, so this must be
	 * paired with set_page_refcounted() (marie_defrag_prep_allocated); a
	 * refcount-0 dst here would be freed mid-migration by folio_add_lru()
	 * (bad_page + live-anon corruption). See defrag_compat.h.
	 */
	marie_defrag_prep_allocated(freepage, order);
	if (order)
		prep_compound_page(freepage, order);

	/*
	 * Age neutrality: pair this dst with its source's gen (looked up from
	 * @srcmap by the src pfn). After migration installs the dst at head,
	 * marie_defrag_topn moves it back to that gen (see restamp_target), so
	 * the relocation does not rejuvenate it. A dst whose migration later
	 * fails is freed without ever being installed, so its byte is not TRACKED
	 * and the restamp simply skips it; a source missing from @srcmap yields
	 * MARIE_DEFRAG_GEN_NONE and falls back to oldest.
	 */
	if (fc->restamp && fc->restamp_n < fc->cap) {
		fc->restamp[fc->restamp_n].pfn = (u32)page_to_pfn(freepage);
		fc->restamp[fc->restamp_n].gen =
			marie_defrag_srcmap_gen(fc, folio_pfn(src));
		fc->restamp_n++;
		atomic_long_inc(&marie_defrag_tot_dst);
	}

	return page_rmappable_folio(freepage);
}

/* put_new_folio: return an unused target to the pool. Mirrors compaction_free(). */
static void marie_defrag_free_target(struct folio *dst, unsigned long data)
{
	struct marie_defrag_freectx *fc = (struct marie_defrag_freectx *)data;
	int order = folio_order(dst);
	struct page *page = &dst->page;

	if (folio_put_testzero(dst)) {
		free_pages_prepare(page, order);
		list_add(&page->lru, &fc->freepages[order]);
		fc->nr += 1UL << order;
	}
}

/* Return leftover (unused) pool pages to the buddy. Mirrors release_free_list(). */
static void marie_defrag_freectx_release(struct marie_defrag_freectx *fc)
{
	int order;

	for (order = 0; order < NR_PAGE_ORDERS; order++) {
		struct page *page, *next;

		list_for_each_entry_safe(page, next, &fc->freepages[order], lru) {
			list_del(&page->lru);
			/*
			 * Mirror release_free_list()/mark_allocated(): __free_pages()'s
			 * put_page_testzero() needs the refcount at 1 to actually free,
			 * so prep the page the same way as a handed-out target (else the
			 * pool leaks on 6.14+). See defrag_compat.h.
			 */
			marie_defrag_prep_allocated(page, order);
			__free_pages(page, order);
		}
	}
	fc->nr = 0;
	/* Scratch is pre-allocated global state; nothing to free here. */
}

/*
 * Harvest sub-pageblock free holes from one block into the pool (until the pool
 * reaches @want). Mirrors isolate_freepages_block: under zone->lock, small
 * PageBuddy free runs are removed via __isolate_free_page. Free runs larger
 * than @fc->harvest_max_order are skipped (marie_defrag_harvest_gate: the
 * requested order minus one, mirroring stock's suitable_migration_target), so a
 * free block big enough to BE the answer is never broken up to serve as targets
 * -- the whole point of the smart-target step.
 */
static void marie_defrag_harvest_block(struct marie_defrag_freectx *fc, unsigned long blk,
			      unsigned long want)
{
	unsigned long start = blk << pageblock_order;
	unsigned long pfn, end = start + (1UL << pageblock_order);
	struct zone *zone;
	unsigned long flags;

	if (fc->nr >= want || !pfn_valid(start))
		return;
	zone = page_zone(pfn_to_page(start));
	/*
	 * Gate the whole pageblock exactly as stock isolate_freepages() does via
	 * pageblock_pfn_to_page(): it rejects an offline/invalid start or end
	 * and -- the corruption guard -- a block that STRADDLES a zone boundary
	 * (a zone may end mid-pageblock). __isolate_free_page() below derives the
	 * target from page_zone(page) INTERNALLY, so every page we hand it under
	 * @zone->lock must belong to @zone; a straddling page would corrupt the
	 * neighbouring zone's free_area unlocked. @zone is only a tentative guess
	 * here (an offline start could make it stale), but pageblock_pfn_to_page()
	 * re-derives via pfn_to_online_page() and bails on any mismatch.
	 */
	if (!pageblock_pfn_to_page(start, end, zone))
		return;

	spin_lock_irqsave(&zone->lock, flags);
	for (pfn = start; pfn < end && fc->nr < want; pfn++) {
		struct page *page = pfn_to_page(pfn);
		unsigned int order;

		if (!PageBuddy(page))
			continue;
		order = buddy_order(page);
		if (order > fc->harvest_max_order ||
		    !__isolate_free_page(page, order)) {
			pfn += (1UL << order) - 1;	/* skip this free run */
			continue;
		}
		set_page_private(page, order);
		list_add_tail(&page->lru, &fc->freepages[order]);
		fc->nr += 1UL << order;
		fc->harvested += 1UL << order;
		pfn += (1UL << order) - 1;
	}
	spin_unlock_irqrestore(&zone->lock, flags);
}

/*
 * Refill the target pool by scanning FORWARD from fc->scan_cursor for more
 * partial movable blocks to harvest holes from (compaction's isolate_freepages
 * analogue) until the pool holds @min_pages or the blocks are exhausted. occ==0
 * blocks are skipped: that covers genuinely-free blocks AND just-emptied source
 * blocks, whose own free pages must stay free to coalesce into the freed
 * order-9 rather than be re-consumed as targets.
 */
static void marie_defrag_harvest_refill(struct marie_defrag_freectx *fc, unsigned long min_pages)
{
	while (fc->nr < min_pages && fc->scan_cursor < marie_defrag_nr_blocks) {
		unsigned long b = fc->scan_cursor++;
		unsigned long pfn = b << pageblock_order;

		if (marie_defrag_block_occupancy(&marie_defrag_hist[b]) == 0)
			continue;
		if (!pfn_valid(pfn) ||
		    get_pageblock_migratetype(pfn_to_page(pfn)) != MIGRATE_MOVABLE)
			continue;
		marie_defrag_harvest_block(fc, b, min_pages);
	}
}

/*
 * Isolate every tracked movable folio of one block into @movelist (the standard
 * lock-free folio_try_get + folio_isolate_lru; isolate fails -- skipped -- for a
 * folio another path already claimed, incl. a Marie folio mid-reclaim). Returns
 * the count isolated.
 */
/*
 * A cold-dead, clean, unmapped FILE folio is DROPPED (reclaimed) rather than
 * moved: near-free -- no target page, no copy, no TLB. @s
 * (marie_state[pfn]) is read just before isolation, so it still carries the
 * gen this folio had at that point. Conservative gate: only the single
 * oldest still-occupied file gen
 * (@oldest_file, K=1 -- marie_find_oldest_occupied_mlv(FILE), NOT a static
 * head-relative age band: that formula assumed a fully-populated ring and
 * under-fired -- never dropping anything -- whenever file occupancy was
 * sparse), and only clean + unmapped + file -- exactly the folios Marie's own
 * swappiness=1 reclaim would drop first, so the refault risk is the normal
 * cold-cache one. @oldest_file < 0 means no file gen is occupied at all,
 * so nothing is droppable. marie_find_oldest_occupied_mlv scans forward from
 * head+1 and so can never return head itself -- the head's own (youngest)
 * gen is structurally excluded from this test, never just a numeric
 * near-miss. anon is never dropped (that would need swap, defeating
 * "near-free"); a per-gen learned p_refault crossover would be a later
 * refinement of this K=1 rule.
 */
static bool marie_defrag_droppable(struct folio *folio, u8 s, int oldest_file)
{
	int gen;

	if (oldest_file < 0)
		return false;
	if (!(s & MARIE_PFN_TYPE_FILE))
		return false;
	if (folio_test_dirty(folio) || folio_test_writeback(folio))
		return false;
	if (folio_mapped(folio) || folio_test_unevictable(folio))
		return false;
	gen = (s & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	return gen == oldest_file;
}

/*
 * How many pages of clean file cache Marie defrag may DROP before breaching the
 * clean_min_ratio floor -- the same reserve Marie's reclaim protects via
 * marie_file_floor_protect (NR_FILE_DIRTY excluded: dirty pages are not
 * droppable and must not count toward the reserve). LONG_MAX when the floor is
 * disabled (ratio 0). Used as a running budget so Marie defrag drops cold file down to,
 * but never below, the floor; folios past the budget fall through to MOVE.
 *
 * Desktop/global-only: computed for the first online node. (NUMA-per-node
 * budgeting is a refinement; Marie's own floor is likewise node-scoped.)
 */
static long marie_defrag_drop_budget(struct pglist_data *pgdat)
{
	unsigned int ratio = READ_ONCE(marie_clean_min_ratio);
	unsigned long file, dirty, file_min;

	if (!ratio)
		return LONG_MAX;
	file = node_page_state(pgdat, NR_ACTIVE_FILE) +
	       node_page_state(pgdat, NR_INACTIVE_FILE);
	dirty = node_page_state(pgdat, NR_FILE_DIRTY);
	file = file > dirty ? file - dirty : 0;
	file_min = pgdat->node_present_pages * ratio / 100;
	return file > file_min ? (long)(file - file_min) : 0;
}

/*
 * Isolate every tracked movable folio of one block, classifying each as a DROP
 * (cold-dead clean file -> @drop_list) or a MOVE (-> @move_list). The standard
 * lock-free folio_try_get + folio_isolate_lru; isolate fails (skipped) for a
 * folio another path already claimed. The droppable test reads the per-PFN byte
 * BEFORE folio_isolate_lru clears it.
 */
/*
 * A folio that cannot be isolated is counted (iso_fail) and skipped. It used to
 * abort the whole block on the argument that a pageblock frees only when all of
 * it is gone, so one folio left behind makes the other five hundred migrations
 * waste. The argument is sound; it was aimed at the wrong stage. Isolation has
 * never lost a folio here -- iso_fail stayed at zero across ~32 million of them
 * -- while migration was failing on half, which is what the 1420-folios-per-
 * block measurement had actually been showing. That was target starvation
 * (marie_defrag_harvest_gate) and is fixed at the source.
 */
static void marie_defrag_isolate_block(struct marie_defrag_freectx *fc, unsigned long blk,
			      struct list_head *move_list,
			      struct list_head *drop_list, int oldest_file,
			      bool may_drop, long *drop_budget,
			      int *n_move, int *n_drop)
{
	unsigned long start = blk << pageblock_order;
	unsigned long pfn, end = start + (1UL << pageblock_order);

	if (end > marie_state_size)
		end = marie_state_size;

	for (pfn = start; pfn < end; pfn++) {
		struct folio *folio;
		long nr;
		bool drop;
		u8 s = READ_ONCE(marie_state[pfn]);

		if (!(s & MARIE_PFN_TRACKED) || !pfn_valid(pfn))
			continue;
		folio = pfn_folio(pfn);
		nr = folio_nr_pages(folio);
		if (folio_test_large(folio) && nr >= (1UL << pageblock_order))
			continue;	/* compound fills block: already contiguous */
		if (!folio_try_get(folio)) {
			atomic_long_inc(&marie_defrag_tot_iso_fail);
			continue;
		}
		/*
		 * @may_drop reflects the request's urgency. Urgent/direct
		 * compaction (an allocation is blocked) drops cold-dead clean
		 * file to free a block with no target -- a refault is cheaper
		 * than the allocation failing. Background/proactive compaction
		 * has no blocked requester, so it MOVES everything and sacrifices
		 * no cache. @drop_budget caps drops at the clean_min_ratio floor;
		 * a droppable folio past the budget falls through to MOVE, so the
		 * protected clean-file reserve is never breached.
		 */
		drop = may_drop && *drop_budget >= nr &&
		       marie_defrag_droppable(folio, s, oldest_file);
		if (!folio_isolate_lru(folio)) {
			/* Lost the folio to a concurrent reclaimer: it is off the
			 * LRU already. This is what leaves a block partially
			 * evacuated, so count it rather than inferring it. */
			atomic_long_inc(&marie_defrag_tot_iso_fail);
			folio_put(folio);
			continue;
		}
		{
			if (drop) {
				list_add(&folio->lru, drop_list);
				(*n_drop)++;
				*drop_budget -= nr;
			} else {
				list_add(&folio->lru, move_list);
				(*n_move)++;
				/*
				 * Age neutrality: record this source's
				 * (pfn, gen) so its migrated dst is restamped
				 * back to this gen. @s was read above, before
				 * folio_isolate_lru cleared the byte, and
				 * folio_pfn (the head pfn) is the key
				 * alloc_target looks up by bsearch.
				 *
				 * MOVE only. A dropped folio is never migrated,
				 * so alloc_target never looks it up; recording
				 * it only lengthens the array this run has to
				 * sort and search.
				 */
				if (fc->srcmap && fc->srcmap_n < fc->cap) {
					fc->srcmap[fc->srcmap_n].pfn =
						(u32)folio_pfn(folio);
					fc->srcmap[fc->srcmap_n].gen =
						(s & MARIE_PFN_GEN_MASK) >>
							MARIE_PFN_GEN_SHIFT;
					fc->srcmap_n++;
				}
			}
		}
		folio_put(folio);
	}
}

/*
 * Put each (pfn, gen) in @tab back to that generation.
 *
 * Used for two things that are the same operation: restoring a migrated
 * destination to its source's age, and restoring a source Marie had to hand
 * back. Both start out sitting at the head gen -- folio_add_lru and
 * folio_putback_lru both route into marie_folio_install, which installs at
 * marie_head_gen -- and both must not keep that youth, because neither a
 * relocation nor a withdrawn attempt is an access.
 *
 * head/oldest are snapshot once so the arc test sees one consistent frame.
 * Entries whose byte is not TRACKED are skipped: a failed migration destination
 * was returned to the pool and never installed.
 */
static unsigned int marie_defrag_apply_restamp(const struct marie_defrag_pg *tab,
					       unsigned int n)
{
	u8 head[ANON_AND_FILE];
	int oldest[ANON_AND_FILE];
	unsigned int j, done = 0;
	int t;

	if (!tab)
		return 0;

	/*
	 * Publish the pending installs first, or most of this is a no-op.
	 *
	 * Everything this function has to restamp arrives on the LRU through
	 * folio_add_lru() (a migration destination) or folio_putback_lru() (a
	 * source handed back by an abort), and both only QUEUE the folio:
	 * __folio_batch_add_and_move() moves the per-CPU batch out to the LRU --
	 * and therefore into Marie via lru_marie_add_folio -> marie_folio_install
	 * -- when the batch fills, not when the folio is added. Until that
	 * happens the PFN's state byte is not TRACKED, and the loop below skips
	 * it as "a failed destination that was never installed". It then gets
	 * installed at the HEAD generation, which is exactly the unearned youth
	 * the restamp exists to undo.
	 *
	 * The blocks this picks are nearly empty by construction (the cost
	 * scorer ranks ascending), so a block is often a handful of folios and
	 * the 15-slot batch rarely fills on its own -- the miss is not a rare
	 * tail. Measured before this drain: 1422 restamps against 1860 migrated
	 * pages, i.e. somewhere between 76% and 100% applied, and not pinnable
	 * more precisely because destinations were not counted. They are now
	 * (marie_defrag_tot_dst), so `restamped` vs `dst` reads directly.
	 *
	 * Process context (kcompactd), no locks held.
	 */
	lru_add_drain();

	for (t = 0; t < ANON_AND_FILE; t++) {
		head[t] = (u8)atomic_read(&marie_head_gen[t]);
		oldest[t] = marie_find_oldest_occupied_mlv(t);
	}
	for (j = 0; j < n; j++) {
		unsigned long pfn = tab[j].pfn;
		int type, target;
		u8 st;

		if (pfn >= marie_state_size)
			continue;
		st = READ_ONCE(marie_state[pfn]);
		if (!(st & MARIE_PFN_TRACKED))
			continue;
		type = (st & MARIE_PFN_TYPE_FILE) ? 1 : 0;
		if (oldest[type] < 0)
			continue;	/* only head occupied: nowhere older */
		target = marie_defrag_restamp_target(tab[j].gen, head[type],
						     oldest[type]);
		marie_state_move_to_gen(pfn, (u8)target);
		done++;
	}
	return done;
}

/*
 * What this run is trying to achieve.
 *
 * @order == 0 means the proactive path: nobody is blocked, there is no specific
 * block to produce, and the caller judges progress by the node's fragmentation
 * score afterwards. The budget is the only bound there.
 *
 * @order > 0 is the demand path: a high-order allocation failed and woke
 * kcompactd. One suitable free block ends the emergency; every block evacuated
 * after that is migration -- and, on this path, potentially DROPped page cache
 * -- spent on nobody's behalf.
 */
struct marie_defrag_req {
	struct pglist_data	*pgdat;
	unsigned int		order;
	enum zone_type		highest_zoneidx;
};

/*
 * Is the blocked allocation satisfiable now? Mirrors the check stock kcompactd
 * makes after compact_zone() (zone_watermark_ok at cc.order against the low
 * watermark), so "done" means the same thing here as it does there.
 */
static bool marie_defrag_target_met(const struct marie_defrag_req *req)
{
	int zoneid;

	if (!req || !req->order)
		return false;

	for (zoneid = 0; zoneid <= req->highest_zoneidx; zoneid++) {
		struct zone *zone = &req->pgdat->node_zones[zoneid];

		if (!populated_zone(zone))
			continue;
		if (zone_watermark_ok(zone, req->order, low_wmark_pages(zone),
				      req->highest_zoneidx, 0))
			return true;
	}
	return false;
}

/*
 * Compact the @n cheapest movable source blocks (capped at MARIE_DEFRAG_TOPK): isolate
 * all their folios, harvest an equal number of target holes from OTHER partial
 * movable blocks, then migrate into the harvested pool.
 *
 * @drop_need is the caller's marie_defrag_drop_need() result: 0 means DROP is
 * off for this run (MOVE only), otherwise the number of pages this specific
 * blocked allocation needs. The actual drop_budget is capped at that need --
 * see marie_defrag_drop_need() -- and separately at the clean_min_ratio floor
 * (marie_defrag_drop_budget()), whichever is smaller.
 */
static void marie_defrag_topn(unsigned int n, unsigned long drop_need,
			      const struct marie_defrag_req *req)
{
	/* Single-owner (marie_defrag_busy) -- see MARIE_DEFRAG_CANDK. */
	static struct marie_defrag_cand top[MARIE_DEFRAG_CANDK];
	struct marie_defrag_freectx fc;
	LIST_HEAD(move_list);
	LIST_HEAD(drop_list);
	int oldest_file;
	bool may_drop = drop_need > 0;
	long drop_budget;
	unsigned int blocks = 0, migrated = 0, dropped = 0, freed = 0, restamped = 0;
	int collected, i, n_move = 0, n_drop = 0, nr_succeeded = 0;
	int nr_drop_here = 0;
	unsigned int rejected_here = 0;

	if (!marie_defrag_hist)
		return;
	/*
	 * The demand path hands down the order it was woken for. Ask ONCE,
	 * before doing any work, whether that request is already satisfiable.
	 *
	 * This used to be re-asked before every block, which reads like "stop as
	 * soon as the caller is served" but is not what it does.
	 * zone_watermark_ok() at @order is satisfied by a SINGLE free block of
	 * that order -- that is the allocator's minimum, not a stock -- so the
	 * run quit after producing one. Measured on a 30 GiB desktop under
	 * sustained THP load: 121 runs, 120 blocks, one per run, against a
	 * demand of 26.6 THP allocations per second. One block is consumed in
	 * under 40 ms, and the next allocation that misses wakes kswapd, which
	 * costs a full Sum(high_wmark) of page cache. Producing exactly the
	 * allocator's minimum is the most expensive stopping rule available.
	 *
	 * No watermark expresses "a stock of N blocks", and inventing an N is
	 * the kind of unmeasured constant this file has been burned by before.
	 * So the budget @n is the only bound past this point -- which is already
	 * how the proactive path behaves, since it passes order 0 and the check
	 * below therefore never applied to it. Turning the demand path's budget
	 * loose the same way moved ~26x more folios per unit time on the same
	 * machine.
	 */
	if (marie_defrag_target_met(req))
		return;
	collected = marie_defrag_collect(top, NULL, NULL);
	if (!collected)
		return;
	/*
	 * @n bounds how many blocks are EVACUATED, not how many are considered:
	 * verification below rejects candidates, and clamping to @collected here
	 * would spend the whole budget on the rejects.
	 *
	 * The clamp stays even though the only caller now passes TOPK itself: it
	 * is what ties @n to the scratch sizing, since the harvest pool and the
	 * srcmap/restamp tables are all allocated for TOPK blocks' worth.
	 */
	if (n > MARIE_DEFRAG_TOPK)
		n = MARIE_DEFRAG_TOPK;
	atomic_long_inc(&marie_defrag_tot_runs);

	/*
	 * Borrow the pre-allocated age-neutrality scratch (no per-run alloc on
	 * this under-pressure path). isolation fills @srcmap, alloc_target reads
	 * it and fills @restamp; the tail applies each restamp.
	 */
	marie_defrag_freectx_init(&fc, req ? req->order : 0);

	/*
	 * Isolate every source folio, splitting into DROP (cold-dead clean file)
	 * and MOVE. This already decremented the source histograms, so the source
	 * blocks read occupancy 0 below.
	 */
	oldest_file = marie_find_oldest_occupied_mlv(1);
	drop_budget = may_drop ?
		min_t(long, (long)drop_need,
		      marie_defrag_drop_budget(NODE_DATA(first_online_node))) : 0;
	/*
	 * Verify before committing. The scorer ranks by evacuation cost, which
	 * says nothing about whether the block can be emptied at all -- see
	 * marie_defrag_skip. Walk the ranked candidates and take the first @n
	 * that a full evacuation would actually free; remember the rest so the
	 * next run does not rank them first all over again.
	 *
	 * @collected can exceed @n, so a run whose cheapest candidates are all
	 * pinned still has deeper ones to fall through to instead of doing
	 * nothing.
	 */
	/*
	 * One block at a time: isolate it, migrate it, restamp it, account it,
	 * then ask whether the allocation that woke us can proceed. The batch
	 * used to be evacuated wholesale and migrated once, which made an early
	 * exit impossible -- a mid-loop watermark test could not see this run's
	 * own progress, because none of it had happened yet. So @n stops being
	 * a target and becomes what it always should have been: a ceiling.
	 *
	 * Everything the refcount-corruption fix depends on is deliberately
	 * outside this loop and untouched: freectx_init/release run exactly once
	 * around it, and every target page still reaches migrate_pages through
	 * marie_defrag_alloc_target -> marie_defrag_prep_allocated (defrag_compat.h,
	 * which is what conditionally adds set_page_refcounted from 6.14 on).
	 * Only srcmap_n/restamp_n are reset per block -- they index per-migration
	 * scratch, not the page pool.
	 */
	for (i = 0; i < collected && blocks < n; i++) {
		int nr_moved = 0;

		/*
		 * Two ways a candidate is worthless, both permanent enough to
		 * remember: nothing here can be freed at all, or there is
		 * nothing here to move in the first place.
		 */
		if (!marie_defrag_block_freeable(top[i].blk)) {
			marie_defrag_skip_mark(top[i].blk);
			atomic_long_inc(&marie_defrag_tot_rej_pinned);
			rejected_here++;
			cond_resched();
			continue;
		}
		if (!marie_defrag_block_has_evacuable(top[i].blk)) {
			marie_defrag_skip_mark(top[i].blk);
			atomic_long_inc(&marie_defrag_tot_rej_nothing);
			rejected_here++;
			cond_resched();
			continue;
		}

		/* Per-migration scratch only; the harvested pool persists. */
		fc.srcmap_n = fc.restamp_n = 0;
		nr_succeeded = 0;

		nr_drop_here = 0;
		marie_defrag_isolate_block(&fc, top[i].blk, &move_list,
					   &drop_list, oldest_file, may_drop,
					   &drop_budget, &nr_moved,
					   &nr_drop_here);
		n_move += nr_moved;
		n_drop += nr_drop_here;
		if (!nr_moved && !nr_drop_here) {
			cond_resched();
			continue;	/* nothing isolatable; not this block's turn */
		}
		blocks++;

		if (fc.srcmap && fc.srcmap_n)
			sort(fc.srcmap, fc.srcmap_n, sizeof(*fc.srcmap),
			     marie_defrag_pg_cmp, NULL);

		/*
		 * Targets reach migrate_pages() only via marie_defrag_alloc_target,
		 * which prepares each harvested page through
		 * marie_defrag_prep_allocated() (defrag_compat.h). If you are porting
		 * this to another kernel, read that helper's PORTING notes before
		 * touching anything here: the refcount contract it encodes changed in
		 * 6.14, and getting it wrong corrupts live pages instead of failing to
		 * build.
		 */
		if (!list_empty(&move_list)) {
			migrate_pages(&move_list, marie_defrag_alloc_target,
				      marie_defrag_free_target,
				      (unsigned long)&fc, MIGRATE_SYNC,
				      MR_COMPACTION, &nr_succeeded);
			migrated += nr_succeeded;
			if (!list_empty(&move_list))
				putback_movable_pages(&move_list);
		}

		/* Age neutrality for this block's destinations. */
		restamped += marie_defrag_apply_restamp(fc.restamp, fc.restamp_n);

		/*
		 * DROP: reclaim the cold-dead clean file folios of THIS block.
		 * reclaim_pages() drops the clean ones (no I/O, no target) and puts
		 * back any that resisted. The source-histogram decrement already
		 * happened at isolation, so a dropped folio simply never re-installs.
		 */
		if (!list_empty(&drop_list))
			dropped += reclaim_pages(&drop_list);

		/*
		 * Outcome for this block. Three cases, only one worth remembering:
		 *
		 *   occ > 0            migration put folios back (target starvation,
		 *                      a locked page, writeback). Still a legitimate
		 *                      candidate -- marking it would exclude a good
		 *                      block over a transient failure.
		 *
		 *   occ == 0, free     what we came for.
		 *
		 *   occ == 0, not free Marie did everything it can here and the block
		 *                      is still pinned by something it does not track:
		 *                      zsmalloc/zram, a balloon page, an orphaned LRU
		 *                      folio. The verification predicate cannot see
		 *                      this class -- has_unmovable_pages() calls those
		 *                      movable, and they are, just not by us. Learn it
		 *                      from the outcome rather than trying to
		 *                      enumerate every such page type and missing one.
		 */
		atomic_long_inc(&marie_defrag_tot_evac);
		if (marie_defrag_block_occupancy(&marie_defrag_hist[top[i].blk])) {
			atomic_long_inc(&marie_defrag_tot_left_occ);
		} else if (marie_defrag_block_is_free(top[i].blk)) {
			freed++;
		} else {
			atomic_long_inc(&marie_defrag_tot_left_pin);
			marie_defrag_skip_mark(top[i].blk);
		}

		cond_resched();
	}

	/*
	 * Amnesty only when the shortfall came from what we REMEMBER, never when
	 * it came from what we just LEARNED.
	 *
	 * A run that rejected candidates has fresh, correct knowledge -- clearing
	 * the map then throws it away and guarantees the next run re-examines the
	 * same losers, which is the very "no memory" failure the map exists to
	 * end. Observed on a 26 GiB page cache: one run rejected all 128
	 * candidates, cleared the map, and would have rejected the same 128
	 * forever. Coming up short with nothing rejected is the opposite case:
	 * the map itself is what stood between us and a full budget, so it is
	 * worth re-examining.
	 */
	if (blocks < n && !rejected_here)
		marie_defrag_skip_amnesty();

	marie_defrag_freectx_release(&fc);

	/* Cumulative diagnostics (read via .../defrag_stats when its flag is on). */
	atomic_long_add(n_move, &marie_defrag_tot_move);
	atomic_long_add(migrated, &marie_defrag_tot_migrated);
	atomic_long_add(dropped, &marie_defrag_tot_dropped);
	atomic_long_add(freed, &marie_defrag_tot_freed);
	atomic_long_add(restamped, &marie_defrag_tot_restamped);
}

/*
 * /sys/kernel/mm/lru_marie/defrag_stats -- A/B / debug node.
 *   write 0/1  : the debug flag (default 0). 1 enables the diagnostic read.
 *   read       : while the flag is on, the cumulative defrag counters (runs,
 *                folios moved/migrated/dropped, blocks freed, dsts restamped,
 *                kcompactd replace fires) plus the histogram completeness
 *                invariant (Sigma_blocks hist == gen_occupied). Off => "disabled".
 *
 * For an A/B vs stock compaction: enable, read totals, run the workload window,
 * read again, diff -- compare migrated/freed against the kernel's compact_*
 * vmstat from a defrag=0 run. No per-run dmesg, no manual trigger.
 */
static ssize_t marie_defrag_stats_show(struct kobject *kobj, struct kobj_attribute *attr,
			    char *buf)
{
	int len, o;

	if (!READ_ONCE(marie_defrag_stats))
		return sysfs_emit(buf, "disabled\n");

	len = sysfs_emit(buf,
		"runs %ld move %ld migrated %ld dropped %ld freed %ld restamped %ld fires %ld\n"
		"rej_pinned %ld rej_nothing %ld skipped %ld amnesty %ld\n"
		"evac %ld freed_ok %ld left_occ %ld left_pin %ld iso_fail %ld\n"
		"dst %ld\n",
		atomic_long_read(&marie_defrag_tot_runs),
		atomic_long_read(&marie_defrag_tot_move),
		atomic_long_read(&marie_defrag_tot_migrated),
		atomic_long_read(&marie_defrag_tot_dropped),
		atomic_long_read(&marie_defrag_tot_freed),
		atomic_long_read(&marie_defrag_tot_restamped),
		atomic_long_read(&marie_defrag_fires),
		atomic_long_read(&marie_defrag_tot_rej_pinned),
		atomic_long_read(&marie_defrag_tot_rej_nothing),
		atomic_long_read(&marie_defrag_tot_skipped),
		atomic_long_read(&marie_defrag_tot_amnesty),
		atomic_long_read(&marie_defrag_tot_evac),
		atomic_long_read(&marie_defrag_tot_freed),
		atomic_long_read(&marie_defrag_tot_left_occ),
		atomic_long_read(&marie_defrag_tot_left_pin),
		atomic_long_read(&marie_defrag_tot_iso_fail),
		atomic_long_read(&marie_defrag_tot_dst));
	/*
	 * One column per source order, so "the pool had nothing above the
	 * harvest gate" is distinguishable from "the pool was simply empty".
	 */
	len += sysfs_emit_at(buf, len, "alloc_fail");
	for (o = 0; o < NR_PAGE_ORDERS; o++)
		len += sysfs_emit_at(buf, len, " %ld",
				     atomic_long_read(&marie_defrag_tot_alloc_fail[o]));
	len += sysfs_emit_at(buf, len, "\n");
	len += sysfs_emit_at(buf, len, "nr_blocks %lu\n", marie_defrag_nr_blocks);
	if (marie_defrag_hist)
		len = marie_defrag_invariant_report(buf, len, PAGE_SIZE);
	return len;
}

static ssize_t marie_defrag_stats_store(struct kobject *kobj, struct kobj_attribute *attr,
			     const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	WRITE_ONCE(marie_defrag_stats, !!v);
	return count;
}

static struct kobj_attribute marie_defrag_stats_attr =
	__ATTR(defrag_stats, 0644, marie_defrag_stats_show, marie_defrag_stats_store);

/*
 * ---------------------------------------------------------------------
 * Master switch + kcompactd replacement.
 * ---------------------------------------------------------------------
 *
 * /sys/kernel/mm/lru_marie/defrag selects WHO performs kcompactd's defrag:
 *   0           -- stock kernel compaction; Marie dormant.
 *   1 (default) -- Marie defrag REPLACES stock compaction on BOTH kcompactd
 *                  paths. mm/compaction.c gates each swap on
 *                  lru_marie_defrag_active():
 *                    - demand (kcompactd_do_work; a high-order allocation is
 *                      blocked) -> lru_marie_defrag_pgdat(pgdat, may_drop=true):
 *                      URGENT, so MOVE + DROP cold-dead clean file -- Marie's
 *                      cheapest, highest-value rung (the clean_min_ratio floor
 *                      still bounds DROP).
 *                    - proactive (background timer) -> may_drop=false: MOVE
 *                      only, no DROP -- no allocation is waiting, so sacrificing
 *                      cache would be pure loss. The proactive loop's
 *                      fragmentation_score-progress defer applies to Marie's
 *                      result identically.
 *
 * may_drop is decided by the PATH (urgency), per the design (commit 95e36ac58f):
 * urgent/direct compaction may sacrifice cold cache to free a block target-free;
 * background must not. Runs in the kcompactd kthread context (sleeping allowed,
 * off the allocation hot path). The .../defrag_run node still triggers a manual
 * one-shot (+N urgent / -N background) regardless of the switch, for testing.
 *
 * /sys/kernel/mm/lru_marie/defrag_drop is a separate, finer-grained killswitch:
 * 0 forces every DROP decision to fall back to MOVE regardless of may_drop,
 * without touching the master defrag switch above (see marie_defrag_drop_enabled).
 *
 * DROP additionally requires marie_defrag_drop_need(pgdat) > 0: may_drop=true
 * alone only means the demand path is order-blocked (fragmented), not that
 * free memory is actually short, so it is gated on an order-0 WMARK_LOW check
 * over the zones that can actually serve the request. When it does
 * fire, the DROP budget is capped at that allocation's own size (1 << order),
 * not "drop everything droppable up to the clean_min_ratio floor" -- see
 * marie_defrag_drop_need() for why headroom already free elsewhere isn't
 * credited against the budget.
 */
/*
 * lru_marie_defrag_active - is the .../defrag switch on, i.e. should Marie
 * defrag replace stock compaction in kcompactd? Read by both compaction.c
 * paths. True only when Marie is enabled, the switch is on, and the histogram
 * exists.
 */
bool lru_marie_defrag_active(void)
{
	return lru_marie_enabled() && READ_ONCE(marie_defrag_enabled) &&
	       marie_defrag_hist;
}
EXPORT_SYMBOL_GPL(lru_marie_defrag_active);

/*
 * Is @pgdat under real reclaim pressure right now, as opposed to merely
 * fragmented, for the actual allocation kcompactd is trying to unblock? A
 * demand-path may_drop=true fires purely on allocation ORDER (a high-order
 * block is unavailable) and says nothing about how much free memory actually
 * exists, so that is what this asks -- at order 0, over the zones that can
 * actually serve the request. Asking at kcompactd's own
 * pgdat->kcompactd_max_order instead, as this originally did, only restates
 * the premise: __zone_watermark_ok() returns false for "no free block of that
 * order" as readily as for "not enough free pages", and the former is a given
 * on the demand path. Combined with a loop that returned on the first failing
 * zone -- and an x86_64 ZONE_DMA that fails any check made for a higher
 * highest_zoneidx -- the gate never once closed. This is deliberately NOT
 * shared with
 * thrash_wd_mem_pressured() (mm/oom_kill.c): that watchdog answers a
 * different, order/zone-agnostic question (is the whole system livelocked)
 * with an empirically-tuned raw global free-vs-high-watermark ratio: mixing
 * in zone_watermark_ok()'s per-zone lowmem_reserve/CMA/highatomic exclusions
 * would silently shift that already-incident-tuned threshold.
 *
 * Returns the DROP budget in pages: 0 when every zone clears the check
 * (nothing to drop), else the size of the block this allocation actually
 * needs (1 << order). Deliberately NOT "block size minus existing
 * watermark headroom": a DROPped folio is freed wherever it happened to be
 * cold in whichever candidate block Marie picked, with no guaranteed
 * spatial relation to wherever that headroom already sits, so crediting it
 * against the budget would assume a contiguity DROP cannot promise. The
 * order's own size is the defensible, self-contained amount -- enough to
 * plausibly cover this one allocation, no more.
 */
static unsigned long marie_defrag_drop_need(struct pglist_data *pgdat)
{
	enum zone_type highest_zoneidx = pgdat->kcompactd_highest_zoneidx;
	unsigned int order = pgdat->kcompactd_max_order;
	int zoneid;

	for (zoneid = 0; zoneid <= highest_zoneidx; zoneid++) {
		struct zone *zone = &pgdat->node_zones[zoneid];

		if (!populated_zone(zone))
			continue;

		/*
		 * Skip zones that can never clear the watermark for
		 * @highest_zoneidx no matter how much memory is free. A zone
		 * whose entire managed size is smaller than
		 * (watermark + lowmem_reserve) fails zone_watermark_ok()
		 * unconditionally, so its failure carries no information about
		 * pressure. On x86_64 ZONE_DMA is exactly this: 16 MiB of
		 * managed pages against a multi-GiB ZONE_NORMAL reserve
		 * (measured: 2816 free vs lowmem_reserve 31071). Reading that
		 * as "free memory is short" made this whole gate fire on every
		 * call -- the loop returns on the FIRST failing zone and
		 * zoneid 0 always failed -- so DROP was permanently enabled
		 * rather than pressure-gated.
		 */
		if (zone_managed_pages(zone) <=
		    low_wmark_pages(zone) + zone->lowmem_reserve[highest_zoneidx])
			continue;

		/*
		 * Ask at order 0, not at @order. Whether THIS high-order
		 * request can be met is already answered -- it cannot, which
		 * is why kcompactd was woken and why we are on the demand
		 * path at all -- so re-asking it here only restates the
		 * premise. The question that is still open, and the one this
		 * gate exists to answer, is whether free memory is actually
		 * short, and that is order-agnostic. zone_watermark_ok() at
		 * @order conflated the two: it returns false for "no free
		 * block of that order" as readily as for "not enough free
		 * pages", i.e. it fails on fragmentation alone.
		 */
		if (!zone_watermark_ok(zone, 0, low_wmark_pages(zone),
					highest_zoneidx, 0))
			return 1UL << order;
	}
	return 0;
}

/*
 * Run one Marie defrag pass for @pgdat. @may_drop is set by the caller from the
 * kcompactd path's urgency: demand=true (MOVE + DROP), proactive=false (MOVE only).
 */
void lru_marie_defrag_pgdat(struct pglist_data *pgdat, bool may_drop)
{
	struct marie_defrag_req req;
	unsigned long drop_need;

	if (!marie_defrag_hist)
		return;
	/*
	 * Single owner: the age-neutrality scratch (srcmap/restamp) is shared
	 * pre-allocated state, and topn walks the global hist/gen ring. A
	 * concurrent caller -- NUMA multi-node kcompactd, or a future manual
	 * trigger -- skips; defrag is best-effort and overlapping runs would only
	 * duplicate work. Also caps the latent NUMA double-run.
	 */
	if (atomic_cmpxchg(&marie_defrag_busy, 0, 1) != 0)
		return;
	/*
	 * Global scan (desktop/single-node). On NUMA this over-triggers across
	 * nodes; per-node Marie defrag is a refinement matching Marie's global posture.
	 */
	atomic_long_inc(&marie_defrag_fires);
	drop_need = (may_drop && READ_ONCE(marie_defrag_drop_enabled)) ?
		marie_defrag_drop_need(pgdat) : 0;

	/*
	 * @may_drop distinguishes the two kcompactd paths, and it also tells us
	 * whether there is a specific allocation to satisfy. The demand path was
	 * woken FOR an order, so hand it down: the run stops as soon as that
	 * order is available again, instead of spending the whole batch on work
	 * nobody asked for. The proactive path has no requester -- order 0 leaves
	 * the budget as the only bound, which is what its caller's
	 * fragmentation-score comparison already assumes.
	 */
	req.pgdat = pgdat;
	req.order = may_drop ? READ_ONCE(pgdat->kcompactd_max_order) : 0;
	req.highest_zoneidx = may_drop ?
		READ_ONCE(pgdat->kcompactd_highest_zoneidx) : pgdat->nr_zones - 1;

	marie_defrag_topn(MARIE_DEFRAG_TOPK, drop_need, &req);
	atomic_set(&marie_defrag_busy, 0);
}

static ssize_t marie_defrag_enabled_show(struct kobject *kobj, struct kobj_attribute *attr,
			     char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(marie_defrag_enabled));
}

static ssize_t marie_defrag_enabled_store(struct kobject *kobj, struct kobj_attribute *attr,
			      const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	WRITE_ONCE(marie_defrag_enabled, !!v);
	return count;
}

static struct kobj_attribute marie_defrag_switch_attr =
	__ATTR(defrag, 0644, marie_defrag_enabled_show, marie_defrag_enabled_store);

static ssize_t marie_defrag_drop_enabled_show(struct kobject *kobj, struct kobj_attribute *attr,
				  char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(marie_defrag_drop_enabled));
}

static ssize_t marie_defrag_drop_enabled_store(struct kobject *kobj, struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int v;
	int err = kstrtouint(buf, 10, &v);

	if (err)
		return err;
	WRITE_ONCE(marie_defrag_drop_enabled, !!v);
	return count;
}

static struct kobj_attribute marie_defrag_drop_attr =
	__ATTR(defrag_drop, 0644, marie_defrag_drop_enabled_show, marie_defrag_drop_enabled_store);

int __init marie_defrag_init(struct kobject *parent)
{
	int err = marie_defrag_hist_alloc();

	if (err)
		return err;

	marie_defrag_scratch_alloc();

	err = sysfs_create_file(parent, &marie_defrag_switch_attr.attr);
	if (err)
		pr_warn("failed to create sysfs defrag node: %d\n", err);
	err = sysfs_create_file(parent, &marie_defrag_stats_attr.attr);
	if (err)
		pr_warn("failed to create sysfs defrag_stats node: %d\n", err);
	err = sysfs_create_file(parent, &marie_defrag_drop_attr.attr);
	if (err)
		pr_warn("failed to create sysfs defrag_drop node: %d\n", err);
	return 0;
}
