/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_STATE_H
#define _MM_LRU_MARIE_STATE_H

#include <linux/percpu_counter.h>
#include "bitmap.h"	/* struct marie_bitmap, marie_bm_* */
#include "defrag.h"		/* marie_defrag_hist_inc/dec -- per-pageblock occupancy mirror */

/* Swappiness/pick diagnostics, surfaced in /sys/kernel/mm/lru_marie/stats. */
extern atomic_long_t marie_dbg_pick[5];
extern atomic_long_t marie_dbg_reclaimed[2];

/*
 * Orphaned-bit self-heal count (surfaced in stats): how many times the
 * isolate scan found an L1 bit set for a (type, zone, oldest_gen) whose
 * marie_state[] byte no longer matched -- see
 * marie_state_isolate_scan_l2lock's function comment. Should stay near
 * zero; the ISOLATED gate closes the common cause, leaving only the
 * narrow residual race marie_state_move_to_gen's comment describes.
 */
extern atomic_long_t marie_dbg_orphan_bit[2];

/*
 * Concede-trigger breakdown (surfaced in stats): which term engaged the
 * FILE_THEN_ANON concede-to-anon -- the clean_min_ratio floor, free-level
 * pressure (marie_node_under_pressure), file-refault feedback
 * (marie_file_refaulting), or a memcg no-file-progress pass.
 */
extern atomic_long_t marie_dbg_concede[4];	/* [floor,free,refault,memcg] */

/*
 * Reclaim rotation counters, harvested from shrink_folio_list's reclaim_stat.
 * With the force-reclaim override gone (see marie_state_shrink_lruvec),
 * rotation IS the mechanism that protects a hot folio, so how much of it is
 * happening is worth being able to read.
 *
 * The two fields are NOT equally clean, and the difference decides how each
 * may be read:
 *
 *   [1] ref_keep = stat->nr_ref_keep. PURE. Bumped only under
 *       `case FOLIOREF_KEEP` (vmscan.c), so it is exactly "folios the
 *       reference check rotated instead of reclaiming".
 *
 *   [0] activate = stat->nr_activate[type]. CONTAMINATED. The activate_locked
 *       label has fourteen predecessors and only one of them is
 *       `case FOLIOREF_ACTIVATE`; the rest are writeback, dirty, swap-alloc
 *       failure, unmap failure and mlock. Read it as a trend, never as a
 *       count of reference-driven activations.
 *
 * Note what neither of these is: a reclaim-quality metric. pgsteal/pgscan is
 * not one either for this subsystem -- Marie's folio_check_references returns
 * FOLIOREF_RECLAIM_CLEAN for once-read clean page cache and that still counts
 * as a steal, so the ratio sits at ~1.0 whatever the policy. Refault counters
 * (workingset_refault_*) are the outcome measure.
 */
extern atomic_long_t marie_dbg_second_chance[2][2];	/* [activate,ref_keep][type] */

/*
 * Budget accounting: [0]=need summed at entry, [1]=delivered, [2]=calls,
 * [3]=largest single-call overshoot. See marie_dbg_budget in state_reclaim.c.
 */
extern atomic_long_t marie_dbg_budget[4];

/*
 * Marie per-PFN state — paradigm specification.
 * ======================================================
 *
 * Marie represents every folio's reclaim state in ONE flat per-PFN byte
 * array (marie_state[]) plus one EXACT hierarchical bitmap
 * (marie_track_bm) used purely to accelerate the reclaim scan.
 *
 * This is the second iteration of this design. The first iteration
 * split marie_state[] into three pieces --
 * identity (marie_state), freshness (marie_age), and a coarse, lossy
 * "hint" bitmap (marie_hint) -- specifically to close a walker/putback
 * race (see the ISOLATED section below) without the exact-bitmap
 * bookkeeping the pre-redesign design required. That traded away scan
 * complexity: the hint's per-PFN presence signal was gone, so
 * marie_state_isolate_scan_l2lock had to walk every PFN in a hint-flagged
 * range densely (O(range)) instead of only real candidates (O(occupancy)).
 * A live tail /dev/zero OOM repro (2026-08-10) measured that walk at 63%
 * of ALL system CPU cycles under low match density -- a multi-minute
 * system-wide freeze the coarse hint could not avoid at any granularity
 * setting, since the O(range) vs O(occupancy) gap is structural, not a
 * tuning problem.
 *
 * This iteration reverts the scan-acceleration structure to the
 * pre-redesign exact L1/L2/refcount bitmap (commit 8f873283e5,
 * marie_bitmap in bitmap.h) while KEEPING the ISOLATED gate the first
 * iteration introduced -- the race the whole effort exists to close is
 * closed by ISOLATED (an ownership signal), not by whether the position
 * data happens to be exact or lossy. GEN moves back into marie_state[]
 * (one byte, one array, one CAS per relocation, matching the
 * pre-redesign layout) and TIER is retired entirely: a folio now promotes
 * straight to the head gen on any access, no intermediate "touched once"
 * step. marie_age[] and marie_hint[] no longer exist.
 *
 * Each Marie operation on a folio is one byte read/write at a fixed
 * PFN-indexed offset (plus an L1/L2 bitmap update on gen changes) -- no
 * allocation anywhere in the fault / del / aging fast paths, no
 * linked-list traversal, no per-CPU staging.
 *
 * marie_state[] is sized once at boot to cover totalram_pages PFNs
 * (~2 MB on a 16 GiB box, ~8 MB on 64 GiB) and never grows or shrinks.
 * The 32-bit PFN gate (marie_init's MARIE_MAX_SUPPORTED_PFN check) caps
 * its worst-case size at 4 GiB. See bitmap.h for marie_track_bm's sizing.
 *
 *
 * marie_state[] byte layout
 * -------------------------
 *
 *   bit 7     TRACKED      1 = folio is owned by Marie; 0 = ignore byte
 *   bit 6     ISOLATED     1 = this PFN is exclusively owned by an
 *                          in-flight isolate/putback right now; every
 *                          other marie_state[] GEN-mutator (walker
 *                          promote-on-access, MADV_COLD, MADV_FREE,
 *                          defrag restamp) must check this first and
 *                          decline if set. See the ISOLATED gate
 *                          protocol below.
 *   bit 5     TYPE         1 = file LRU, 0 = anon LRU
 *   bit 4..3  ZONE         folio_zonenum: 0=DMA, 1=DMA32, 2=NORMAL, 3=MOVABLE
 *   bit 2..0  GEN          relative-position 0..7 in the cycling ring
 *                          (0 = oldest, head = marie_head_gen[type])
 *
 * Ordered MSB -> LSB by hierarchy, not by minimal diff from any prior
 * layout: TRACKED and ISOLATED are gates (existence, then ownership) that
 * must be checked before anything else in the byte means anything; TYPE
 * and ZONE are near-immutable identity attributes of the physical page;
 * GEN is the one field that changes on nearly every access or reclaim
 * pass, and so sits at the bits every relocation's shift/mask touches.
 *
 * Untracked PFNs read as 0. TRACKED is the single source of truth for
 * "does Marie own this PFN" -- no separate folio->flags Marie bit is used.
 *
 * Any relocation in gen space (promote-on-access or reclaim putback) is a
 * single atomic CAS on this one byte, immediately followed by a
 * marie_bm_set(new plane) + marie_bm_clear(old plane) pair on
 * marie_track_bm and a matched marie_gen_occupied adjustment -- see
 * marie_state_move_to_gen.
 *
 *
 * ISOLATED gate protocol
 * -----------------------
 *
 * marie_state[pfn]'s GEN field has exactly four mutators besides
 * isolate/putback themselves, all of which MUST check ISOLATED first and
 * decline (no-op) if set:
 *
 *   1. Walker / folio_mark_accessed promote-on-access (marie_state_inc_tier,
 *      name kept for API stability -- see below)
 *   2. MADV_COLD (marie_folio_demote)
 *   3. MADV_FREE (lru_marie_lazyfree)
 *   4. Marie defrag's post-migration restamp (defrag.c)
 *
 * Isolate (marie_evict_counters_only) sets ISOLATED as the very first
 * thing it does, before touching anything else, narrowing the window any
 * of the four mutators above could race in. Putback (state.c, the
 * survivor release path) is the only code that clears ISOLATED, and does
 * so LAST, after GEN/marie_track_bm/marie_gen_occupied are all already
 * consistent -- so no gated mutator ever observes a gate-open window with
 * stale position data.
 *
 * This is the one property that survives unchanged from the first
 * redesign iteration: the race this whole effort exists to close (an
 * uncoordinated walker vs. isolate/putback bit flip producing orphaned
 * marie_track_bm bits and phantom gen_occupied counts) is closed by
 * ISOLATED being an explicit, checkable ownership token -- independent of
 * whether the position data underneath it (marie_track_bm) is exact or
 * lossy.
 *
 *
 * Isolate scan
 * ------------
 *
 * Per (type, zone, gen), marie_track_bm's L1 bit is the exact "is this
 * PFN here" signal: the scanner's __ffs-based extraction visits only real
 * candidates (O(occupancy)), and the L2 512-bit summary (refcounted, also
 * exact) lets it skip whole PFN ranges with zero candidates without
 * touching L1 at all. See marie_state_isolate_scan_l2lock.
 *
 *
 * Aging — gen ring as a cycling counter (per type)
 * ------------------------------------------------
 *
 *   atomic_t marie_head_gen[ANON_AND_FILE];           // 0..NR_GENS-1 cycling per type
 *   atomic_long_t marie_gen_installs[MARIE_PFN_NR_GENS][ANON_AND_FILE];
 *   atomic_long_t marie_gen_occupied[MARIE_PFN_NR_GENS][ANON_AND_FILE];
 *
 * install:
 *
 *   u8 gen = atomic_read(&marie_head_gen[type]);
 *   marie_state[pfn] = MARIE_PFN_TRACKED |
 *                      (type ? MARIE_PFN_TYPE_FILE : 0) |
 *                      marie_pfn_zone_bits(zone) |
 *                      (gen << MARIE_PFN_GEN_SHIFT);
 *   marie_bm_set(&marie_track_bm[type][zone][gen], pfn);
 *   atomic_long_inc(&marie_gen_installs[gen][type]);
 *   atomic_long_inc(&marie_gen_occupied[gen][type]);
 *
 * head_gen advance is global and per-type, drain-wait gated (next gen
 * empty for that type), and fired by install cadence alone: a single
 * global per-type counter (marie_gen_installs[type]) counts installs onto
 * the head gen and advances at marie_gen_growth_threshold(type)
 * (marie_folio_install -> marie_try_advance_head_mlv, under lru_lock). The
 * former reclaim-time "occupied < 2 at shrink entry" trigger was removed --
 * under concurrent global reclaim it thrashed the ring and collapsed age
 * stratification.
 *
 * marie_find_oldest_occupied_mlv() is a live, uncached O(NR_GENS) search
 * every time it is called (by the scan sweep, by putback, by MADV_COLD /
 * MADV_FREE) -- deliberately NOT a maintained head/tail cursor. A cached
 * "tail" would need every gen-repopulating write (MADV_COLD/MADV_FREE
 * target "the current oldest gen" directly, entirely outside the
 * isolate/putback subsystem and asynchronous to it) to keep the cursor
 * correctly in sync, and getting that wrong reintroduces exactly the
 * "position data quietly stops matching reality" bug class this file
 * exists to close. The live search is already O(NR_GENS)=8 atomic reads
 * and was never the measured bottleneck (marie_state_isolate_scan_l2lock
 * was).
 *
 *
 * Del — one byte zero
 * ---------------------
 *
 *   marie_state[pfn] = 0;
 *
 * No swap-pop, no list_del, no shard lock dance. External del
 * (lru_marie_del_folio from compaction, folio_put, munmap) is the
 * same store (plus the matching marie_bm_clear).
 *
 *
 * memcg scope
 * -----------
 *
 * marie_state[] and marie_track_bm are global (single allocation
 * system-wide), not per-memcg. Marie is desktop/global-only: there is no
 * per-memcg reclaim and the scan covers every Marie folio (no per-memcg
 * filter). This trades per-memcg locality for vastly simpler data
 * structures — desktop and small-server cgroup trees (where Marie
 * targets) are dominated by the root memcg anyway, so the locality loss
 * is small in practice.
 *
 *
 * Walker integration
 * ------------------
 *
 * The PTE walker (marie_walker) inspects young bits as before and calls
 * marie_state_inc_tier() on a young PFN (name kept for API stability;
 * the walker's call sites are unchanged). Internally this now promotes
 * the folio straight to the head gen on any access (no intermediate
 * step), after checking marie_state[pfn]'s ISOLATED gate. The same SIMD
 * young-pte machinery from the prior implementation carries over
 * unchanged.
 *
 *
 * Disable
 * -------
 *
 * Marie disable (boot-only configuration): write 0 to every TRACKED
 * byte via SIMD bulk store, folio_put each one. An O(N) sweep but it
 * happens rarely. (Memcg reparent is gone -- desktop/global-only.)
 *
 *
 * Sizing & init
 * -------------
 *
 * marie_state is kvmalloc'd at subsys_initcall with size `max_pfn` bytes.
 * max_pfn is bounded by the 32-bit PFN gate (marie_init's
 * MARIE_MAX_SUPPORTED_PFN check), so the array is at most 4 GiB on the
 * maximum supported config. Realistic sizings:
 *
 *   16 GiB RAM  ->  2 MiB   (single kvmalloc, contiguous in vmalloc)
 *   64 GiB RAM  ->  8 MiB
 *  256 GiB RAM  -> 32 MiB
 *
 * marie_track_bm is 2 types * MARIE_PFN_NR_ZONES_ENCODED(4) zones * 8 gens
 * = 64 planes; see bitmap.h for the per-plane L1/L2/refcount sizing.
 *
 * marie_state is sparse-tolerant: NUMA holes and reserved regions read
 * as 0 (untracked) and incur only sequential-read cost during scan.
 */

/*
 * Field shifts and masks within each marie_state[] byte. Ordered
 * MSB -> LSB by hierarchy (see the byte-layout block above): TRACKED,
 * ISOLATED, TYPE, ZONE, GEN. GEN is built up from bit 0 first since it is
 * the field every relocation's shift/mask actually touches; the others
 * are derived upward from it.
 */
#define MARIE_PFN_GEN_BITS		3
#define MARIE_PFN_GEN_SHIFT		0
#define MARIE_PFN_GEN_MASK		(((1U << MARIE_PFN_GEN_BITS) - 1) << \
					 MARIE_PFN_GEN_SHIFT)
#define MARIE_PFN_NR_GENS		(1U << MARIE_PFN_GEN_BITS)

#define MARIE_PFN_ZONE_SHIFT		(MARIE_PFN_GEN_SHIFT + MARIE_PFN_GEN_BITS)
#define MARIE_PFN_ZONE_BITS		2
#define MARIE_PFN_ZONE_MASK		(((1U << MARIE_PFN_ZONE_BITS) - 1) << \
					 MARIE_PFN_ZONE_SHIFT)
#define MARIE_PFN_NR_ZONES_ENCODED	(1U << MARIE_PFN_ZONE_BITS)

#define MARIE_PFN_TYPE_SHIFT		(MARIE_PFN_ZONE_SHIFT + MARIE_PFN_ZONE_BITS)
#define MARIE_PFN_TYPE_FILE		(1U << MARIE_PFN_TYPE_SHIFT)
#define MARIE_PFN_TYPE_MASK		MARIE_PFN_TYPE_FILE

#define MARIE_PFN_ISOLATED_SHIFT	(MARIE_PFN_TYPE_SHIFT + 1)
#define MARIE_PFN_ISOLATED		(1U << MARIE_PFN_ISOLATED_SHIFT)

#define MARIE_PFN_TRACKED_SHIFT		(MARIE_PFN_ISOLATED_SHIFT + 1)
#define MARIE_PFN_TRACKED		(1U << MARIE_PFN_TRACKED_SHIFT)

/*
 * The 8 gens give a longer head->oldest descent (time-domain grace),
 * which measurably protects warm/marginally-hot folios: a QEMU warm-set
 * A/B (cold filler + a working set re-accessed near the grace period)
 * measured ~40% fewer warm-folio refaults than a former 4-gen split.
 * TIER (a separate 1-bit "touched once" hysteresis before promoting to
 * head) is retired in this iteration: a folio now promotes straight to
 * head on any access. TIER's own A/B was never run (only the gen-count
 * axis was measured above) and single-touch-promotes-to-head is standard
 * behaviour in most classic LRU implementations, not an exotic policy --
 * revisit with measurement if refault regressions show up under a
 * workload with heavy single-touch scan traffic (e.g. updatedb/backup
 * indexing).
 */

/*
 * Encode @zone (folio_zonenum result) into the byte's zone nibble.
 * Truncates to MARIE_PFN_NR_ZONES_ENCODED-1 so ZONE_DEVICE etc. do
 * not overflow the 2-bit field; in practice those zones do not
 * reach Marie's install path.
 */
static inline u8 marie_pfn_zone_bits(unsigned int zone)
{
	return (u8)((zone & (MARIE_PFN_NR_ZONES_ENCODED - 1)) <<
		    MARIE_PFN_ZONE_SHIFT);
}

/* The base allocation (subsys_initcall) covers totalram_pages PFNs. */
extern u8 *marie_state;
extern unsigned long marie_state_size;

/*
 * Per-(gen, type) live folio count, kept as a node-wide population
 * aggregate by the marie_gen_occ_inc/dec abstraction (state.c). Aging is
 * global now (single per-type clock: marie_head_gen / marie_gen_occupied);
 * this sum is the global gen occupancy and also serves as the "is any anon
 * tracked?" signal for marie_file_floor_protect.
 */
extern atomic_long_t marie_gen_occupied[MARIE_PFN_NR_GENS][2 /* ANON_AND_FILE */];

/*
 * How many of this type's generations currently hold anything, maintained
 * incrementally by the marie_gen_occ_inc/dec choke point below rather than
 * derived by scanning all NR_GENS counters.
 *
 * The value only moves when a generation crosses 0 <-> non-zero, which happens
 * a handful of times per lap, so keeping it is near-free while reading it is a
 * single atomic_read -- as opposed to NR_GENS reads of the hottest, most
 * contended cachelines Marie has (every install and every eviction writes
 * marie_gen_occupied). That matters because the install cadence is a hot path
 * and this is the input a ring-occupancy-aware growth threshold would need
 * live rather than sampled once per head advance.
 *
 * Note it counts OCCUPIED generations, which is not the same as the span from
 * the head back to the oldest occupied one: a generation in the middle can
 * empty out when its last folio is promoted to the head or freed, and
 * marie_find_oldest_occupied_mlv skips such holes by design.
 */
extern atomic_t marie_nr_occupied_gens[2 /* ANON_AND_FILE */];

/*
 * gen_occupancy abstraction -- the SINGLE interface for "a folio entered /
 * left gen @g of @type". EVERY per-PFN gen-transition site routes its
 * occupancy bookkeeping through here: the install/split publisher
 * (marie_pfn_publish_inherit, pfn_install.h) and the state.c transitions
 * (publish_at_gen, move_to_gen, inc_tier, drop_pfn, drop_pfn_at_free,
 * evict_counters_only, uncharge_backstop). Keeping it a single choke-point
 * means a new transition site cannot forget to update the counter -- and
 * gives the per-pageblock occupancy mirror (Marie defrag compaction, marie_defrag_hist_inc/dec)
 * one hook to tap, with Sigma_blocks(mirror) == Sigma(marie_gen_occupied) as
 * its completeness invariant. @pfn is threaded through purely so the mirror
 * can locate the folio's pageblock; the marie_gen_occupied bump ignores it.
 * When CONFIG_LRU_MARIE_DEFRAG=n the marie_defrag hooks are empty inlines and the
 * emitted code is identical to a bare atomic_long_inc/dec.
 *
 * static inline in the header (was state.c-private) precisely so the
 * install-side publisher in pfn_install.h shares this path instead of
 * bumping marie_gen_occupied directly. Desktop/global-only: a single global
 * per-(gen, type) counter, so these take no carrier.
 */
static inline void marie_gen_occ_inc(unsigned long pfn, int gen, int type)
{
	/*
	 * *_return rather than a bare inc/dec so the 0 <-> non-zero crossing can
	 * be detected: both are the same locked RMW on the same cacheline (lock
	 * xadd instead of lock inc on x86), so the occupied-generation tally
	 * costs a register add on the common path and one extra atomic on the
	 * rare crossing. Exactly one caller observes each crossing, so the tally
	 * needs no lock of its own.
	 */
	if (atomic_long_inc_return(&marie_gen_occupied[gen][type]) == 1)
		atomic_inc(&marie_nr_occupied_gens[type]);
	marie_defrag_hist_inc(pfn, gen, type);
}

static inline void marie_gen_occ_dec(unsigned long pfn, int gen, int type)
{
	if (atomic_long_dec_return(&marie_gen_occupied[gen][type]) == 0)
		atomic_dec(&marie_nr_occupied_gens[type]);
	marie_defrag_hist_dec(pfn, gen, type);
}

/*
 * marie_acct_pred - does this state byte OCCUPY a slot?
 *
 * The single residency predicate the whole module derives its counters
 * from. A PFN occupies its (type, gen) slot -- and holds an outstanding
 * marie_nr_folios/lru_size credit -- exactly while it is TRACKED and not
 * ISOLATED:
 *
 *   TRACKED=1, ISOLATED=0   resident   (installed, or put back)
 *   TRACKED=1, ISOLATED=1   not        (reclaim isolate owns it)
 *   TRACKED=0               not        (evicted / freed / never in)
 *
 * The two lifecycle bits ARE the ledger; no spare bit is needed (the byte
 * is full: GEN 3 + ZONE 2 + TYPE 1 + ISOLATED 1 + TRACKED 1). Defined here
 * next to the byte layout because both marie_gen_occ_settle() below and
 * account.h's marie_acct_settle_*() are projections of this one predicate:
 * gen_occupied partitions it by (gen, type); marie_nr_folios totals it.
 */
static inline bool marie_acct_pred(u8 b)
{
	return (b & MARIE_PFN_TRACKED) && !(b & MARIE_PFN_ISOLATED);
}

/*
 * marie_gen_occ_settle - move gen_occupied to match a COMMITTED marie_state[]
 * transition @old -> @new_b for @pfn.
 *
 * gen_occupied is a projection of marie_acct_pred() onto (gen, type), so it
 * is derived from the byte -- the single source of truth -- rather than from
 * marie_track_bm's bit transitions.
 *
 * That inversion is the point. While gen_occupied was maintained from
 * marie_bm_set/marie_bm_clear return values, the bitmap had to be a source
 * of truth too, so ANY byte-vs-bitmap disagreement became a corrupted
 * counter: a stale bit the scanner self-healed away decremented occupancy
 * for a folio that was still resident (find_oldest then returns a gen the
 * scanner finds empty -> zero reclaim progress -> premature OOM), and a
 * not-yet-published bit could be "healed" out from under a live folio,
 * stranding it where no future scan would look. Keeping two independently
 * mutated authorities in agreement is what forced the per-PFN-range lock in
 * marie_state_move_to_gen().
 *
 * With occupancy derived from the byte, marie_track_bm is demoted to what it
 * always should have been: a lossy, self-healing SCAN INDEX. A stale bit
 * costs one wasted scanner visit; a momentarily missing bit costs one missed
 * scan. Neither can move a counter, so neither needs a lock -- and the
 * range lock is gone.
 *
 * Exactly-once by construction, like account.h's counter settle: the delta
 * follows from (@old, @new_b), so only the caller whose atomic RMW actually
 * committed the transition computes a nonzero one.
 */
static inline void marie_gen_occ_settle(unsigned long pfn, u8 old, u8 new_b)
{
	bool o = marie_acct_pred(old), n = marie_acct_pred(new_b);
	int og, ng, ot, nt;

	if (!o && !n)
		return;

	og = (old & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	ng = (new_b & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
	ot = (old & MARIE_PFN_TYPE_MASK) ? 1 : 0;
	nt = (new_b & MARIE_PFN_TYPE_MASK) ? 1 : 0;

	/* Same slot on both sides and resident throughout: nothing moved. */
	if (o && n && og == ng && ot == nt)
		return;

	if (o)
		marie_gen_occ_dec(pfn, og, ot);
	if (n)
		marie_gen_occ_inc(pfn, ng, nt);
}

/*
 * marie_state_names_plane - does @b place this PFN in (@type, @zone, @gen)?
 *
 * The same mask/target comparison the scanner makes, so "the index bit for
 * this plane should exist" has exactly one definition.
 */
static inline bool marie_state_names_plane(u8 b, int type, int zone, int gen)
{
	const u8 mask = MARIE_PFN_TRACKED | MARIE_PFN_ISOLATED |
			MARIE_PFN_TYPE_MASK | MARIE_PFN_ZONE_MASK |
			MARIE_PFN_GEN_MASK;
	const u8 want = MARIE_PFN_TRACKED |
			(type ? MARIE_PFN_TYPE_FILE : 0) |
			marie_pfn_zone_bits(zone) |
			((u8)gen << MARIE_PFN_GEN_SHIFT);

	return (b & mask) == want;
}


/* Global aging clock (desktop/global-only); see state.c for the contract. */
extern atomic_t marie_head_gen[2 /* ANON_AND_FILE */];
extern atomic_long_t marie_gen_installs[2 /* ANON_AND_FILE */];

/*
 * Install-cadence threshold, derived per install from a slowly-varying base and
 * the LIVE occupied-generation count. Fully automatic (no sysfs knob). See the
 * block above marie_recompute_growth_base() in state_core.c for why the cadence
 * is a function of ring occupancy at all, and for the equilibrium it settles at.
 */
extern unsigned long marie_gen_growth_base[2 /* ANON_AND_FILE */];
void marie_recompute_growth_base(int type);

/* Extra right-shift on the cadence floor; see state_core.c. 0 = brake only. */
#define MARIE_GEN_FLOOR_SHIFT_MAX \
	(MARIE_PFN_NR_GENS + 1 - const_ilog2(MARIE_PFN_NR_GENS))
extern unsigned int marie_gen_floor_shift;

static inline unsigned long marie_gen_growth_threshold(int type)
{
	int gens = atomic_read(&marie_nr_occupied_gens[type]);
	unsigned long base;

	/*
	 * Clamp before shifting. gens is derived from 0 <-> non-zero crossings of
	 * marie_gen_occupied, so an accounting drift there would carry into it,
	 * and this shift must not be handed a negative or oversized count.
	 * Clamping keeps a miscount to a wrong cadence rather than undefined
	 * behaviour.
	 */
	gens = clamp(gens, 0, (int)MARIE_PFN_NR_GENS);

	base = READ_ONCE(marie_gen_growth_base[type]);

	return max(base >> (MARIE_PFN_NR_GENS + 1 - gens),
		   base >> (const_ilog2(MARIE_PFN_NR_GENS) +
			    READ_ONCE(marie_gen_floor_shift)));
}

/*
 * Oldest live gen for @type (walking out from the global head), or -1 if none.
 * The single placement primitive for "make this folio cold now": demote /
 * lazyfree (state.c) and Marie defrag's post-migration restamp all target it so
 * a folio lands at the current tail, frame-relative -- never a stale absolute
 * gen that head has since recycled. See marie_find_oldest_occupied_mlv.
 */
int marie_find_oldest_occupied_mlv(int type);

/*
 * Try to advance the global head gen for @type by one slot (resets the
 * install counter). Internal to state_core.c/state_reclaim.c/state_folio.c's
 * split across the aging clock, the reclaim driver, and the folio
 * install path -- not part of the module's external API.
 */
bool marie_try_advance_head_mlv(int type);

/* Global anon-vs-file proportional pick bias; see marie_swap_pick_type. */
extern atomic64_t marie_swap_bias;

/*
 * Per-(type, zone, gen) EXACT scan-acceleration bitmap. One struct
 * marie_bitmap per (type, zone, gen) triple -- see bitmap.h. This is a
 * source of truth for "is this PFN here" (unlike the retired marie_hint):
 * a set L1 bit always means the PFN genuinely is tracked at that exact
 * (type, zone, gen), so the scanner's __ffs extraction only ever visits
 * real candidates.
 *
 * Single global per-(type, zone, gen) plane; there is no per-memcg
 * instance (desktop/global-only).
 */
extern struct marie_bitmap marie_track_bm[2 /* ANON_AND_FILE */]
					 [MARIE_PFN_NR_ZONES_ENCODED]
					 [MARIE_PFN_NR_GENS];

/*
 * marie_bm_retire - remove @pfn from ONE scan-index plane, race-free.
 *
 * Returns true iff the bit was genuinely stale and is now gone (the caller
 * may count that as an orphan); false if there was nothing to remove, or if
 * what we removed turned out to be a LIVE publish that we put back.
 *
 * Why a bare marie_bm_clear() is not enough. Clearing is the only index
 * operation that can destroy information, and it is decided from a byte read
 * that is not atomic with it:
 *
 *   1. a retirer reads marie_state[pfn], sees it does NOT name plane P, and
 *      decides P's bit is stale;
 *   2. concurrently a mutator moves this PFN INTO P -- CAS byte to P, then
 *      marie_bm_set(P);
 *   3. the retirer's marie_bm_clear(P) lands last and removes the bit the
 *      mutator just published, while the mutator's own marie_bm_clear of the
 *      PFN's PREVIOUS plane removes the other copy.
 *
 * The byte then names P with no index bit anywhere: the folio is TRACKED and
 * counted in gen_occupied[P], but no future scan can reach it. That is a
 * permanently stranded, unreclaimable folio AND phantom occupancy that makes
 * marie_find_oldest_occupied_mlv keep returning a gen the scanner finds
 * empty (zero reclaim progress, and head advance wedged on its
 * gen_occupied[next]==0 gate) -- premature OOM with memory resident. Both
 * the scanner's stale-bit self-heal and marie_state_move_to_gen's retire of
 * the old plane can hit this; it is not specific to one caller, which is why
 * it is fixed here rather than at a call site.
 *
 * Clear-then-repair closes it without a lock, because the repair is ORDERED
 * AFTER the destructive step. Let the last clear of P for this PFN be at t3
 * and its re-read at t4 > t3. If the settled byte names P, the CAS that
 * wrote it is at some t1 with its paired marie_bm_set(P) at t2 > t1:
 *
 *   - t2 > t3: the mutator's own set lands after our clear. Bit present.
 *   - t2 < t3: then t1 < t2 < t3 < t4, so our re-read at t4 must observe P
 *     (nothing later changed it -- P is the settled value) and we re-set it.
 *
 * If the settled byte does not name P, an absent bit is correct. So in every
 * interleaving the index converges on "the plane the byte names has a bit",
 * which is the property the scanner needs. Cost is one byte read, and rarely
 * one bit set, on the retire path only -- the common "bit was already clear"
 * case pays nothing (a mutator whose marie_bm_set has not run yet will run
 * it after us, so returning early there is safe too).
 */
static inline bool marie_bm_retire(unsigned long pfn, int type, int zone,
				   int gen)
{
	struct marie_bitmap *bm = &marie_track_bm[type][zone][gen];

	if (!marie_bm_clear(bm, pfn))
		return false;

	if (marie_state_names_plane(READ_ONCE(marie_state[pfn]), type, zone,
				    gen)) {
		marie_bm_set(bm, pfn);
		return false;
	}
	return true;
}

/*
 * clean_min_ratio: minimum file-pagecache reserve as percent of
 * node_present_pages. Sysfs-tunable in core.c, read by reclaim.
 * Default 15 (le9uo recommendation for desktop).
 */
extern unsigned int marie_clean_min_ratio;

#include <linux/percpu.h>
/* One-shot init from marie_init(). Allocates marie_state with kvmalloc
 * and marie_track_bm's planes via marie_bm_init(). */
int marie_state_init(void);
/* Detect CPUID-based prefetch ring parameters. Call before marie_state_init(). */
void marie_prefetch_params_init(void);

struct pglist_data;
struct folio;
struct lruvec;
struct scan_control;
struct mem_cgroup;

/*
 * L1/L2-bitmap-accelerated isolate scan for one (type, gen), looping
 * zone = 0..max_zone: per zone, word-ANDs marie_track_bm[type][zone][gen]'s
 * L2 summary to skip empty PFN ranges in one cycle, takes an L2 range
 * under a try_lock for exclusive ownership, then __ffs-extracts only the
 * L1 bits actually set in that range (O(occupancy), not O(range) -- see
 * the ISOLATED-gate-vs-scan-structure note at the top of this file).
 * marie_state[pfn] alone (TRACKED/TYPE/ZONE/GEN/ISOLATED, one read) is
 * enough to confirm each extracted candidate; there is no second array to
 * cross-check. Global-only: the scan always covers every Marie folio (no
 * per-memcg filter).
 */
unsigned long marie_state_isolate_scan_l2lock(struct pglist_data *pgdat,
					      int type, int max_zone,
					      struct folio **batch,
					      unsigned long batch_size,
					      unsigned long nr_to_scan,
					      int oldest_in);

/*
 * Per-PFN-array reclaim driver. Walks @type via
 * marie_state_isolate_scan_l2lock, claims each candidate via
 * try_get + test_clear_lru, hands the resulting folio_list to
 * shrink_folio_list, and putbacks any survivors. Sole reclaim
 * driver in PFN-only Marie.
 */
unsigned int marie_state_shrink_lruvec(struct lruvec *lruvec,
				       struct scan_control *sc);

/*
 * Marie type-pick return codes for marie_swap_pick_type().
 *
 *   MARIE_PICK_FILE_STRICT  swappiness=0:   FILE only, no ANON fallback;
 *                                           caller proceeds to OOM if FILE
 *                                           is depleted.
 *   MARIE_PICK_ANON_STRICT  swappiness=MAX: ANON only, no FILE fallback.
 *   MARIE_PICK_FILE_THEN_ANON  swappiness=1: FILE first; ANON engages
 *                                            ONLY when skip_file is set
 *                                            (clean_min_ratio breached).
 *                                            Per-call transient FILE
 *                                            failures do not promote to
 *                                            ANON -- the floor itself is
 *                                            the sole depletion signal.
 *   MARIE_PICK_ANON_FIRST   Proportional regime (s=2..199), bias picks
 *                           ANON. SINGLE type per call -- scanning the
 *                           other side would dissolve the s:(MAX-s)
 *                           page-flow ratio. Bias gets updated from
 *                           this call's outcome, possibly flipping the
 *                           pick for the next shrink_lruvec call.
 *   MARIE_PICK_FILE_FIRST   Symmetric to ANON_FIRST: bias picks FILE,
 *                           single type per call.
 */
enum marie_pick_kind {
	MARIE_PICK_FILE_STRICT,
	MARIE_PICK_ANON_STRICT,
	MARIE_PICK_FILE_THEN_ANON,
	MARIE_PICK_ANON_FIRST,
	MARIE_PICK_FILE_FIRST,
};

/*
 * Resolve the type-pick policy for one shrink_lruvec invocation.
 *
 * Pure read of the controller state: looks at @swappiness to detect
 * the {0, 1, MAX_SWAPPINESS} special values, otherwise reads the global
 * marie_swap_bias sign to pick the primary type for the proportional
 * regime. Does not modify any state.
 */
enum marie_pick_kind marie_swap_pick_type(u8 swappiness);

/*
 * Apply the bias-controller update for one ATTEMPTED pick.
 *
 *   nr_reclaimed > 0  -> bias += sign * nr_reclaimed * weight
 *                        Page-flow proportional: long-run
 *                        pages(anon):pages(file) -> s:(MAX-s) even
 *                        when per-pick batches differ between types.
 *
 *   nr_reclaimed == 0 -> no-op (bias unchanged)
 *                        Failure carries no back-pressure. The
 *                        picked side stays the picked side
 *                        indefinitely under sustained failure;
 *                        anon is not surrendered just because file
 *                        is transiently or persistently stuck.
 *
 *   sign   = -1 for picked=ANON (push toward FILE)
 *            +1 for picked=FILE (push toward ANON)
 *   weight = MAX_SWAPPINESS - s   for picked=ANON
 *          = s                    for picked=FILE
 *
 * Bypassed entirely under special-value swappiness (0, 1, MAX),
 * where the pick is deterministic and the global bias is not consulted.
 *
 * Caller MUST only invoke when the pick was actually attempted;
 * do NOT call when an external override (skip_file from
 * clean_min_ratio) blocked the picked type before the scan ran.
 */
void marie_swap_bias_update(int picked_type,
			    unsigned long nr_reclaimed,
			    u8 swappiness);

/*
 * Promote-to-head-on-access for a Marie-tracked folio.
 *
 * Name kept from the prior (saturating-tier) design for API stability --
 * walker.c and the folio_mark_accessed() hook call this by name and need
 * no changes. Behaviour is now unconditional: any access moves the folio
 * straight to the head gen (no intermediate "touched once" tier step --
 * see the retirement note in the byte-layout block above).
 *
 * First checks marie_state[pfn]'s ISOLATED bit and declines (no-op,
 * touches nothing) if set -- this PFN is exclusively owned by an
 * in-flight isolate/putback right now (see the ISOLATED gate protocol
 * above). This is a plain check-and-decline, not a retry: there is
 * nothing to retry, the walker simply has no work to do on a folio
 * mid-reclaim.
 *
 * Otherwise delegates to marie_state_move_to_gen() (below) for the actual
 * relocation -- a single atomic CAS on marie_state[pfn] plus the matching
 * marie_track_bm/marie_gen_occupied update.
 *
 * Skips quietly if the folio is not (or no longer) tracked, or is
 * already on the head gen.
 */
void marie_state_inc_tier(unsigned long pfn);

/*
 * marie_state_inc_tier_seeded - promote-on-access from an already-read
 * state byte. Walker-only fast path: the caller has bounds-checked @pfn
 * and read marie_state[pfn] into @cur for the TRACKED gate, so this skips
 * that reload and also checks ISOLATED against the same @cur rather than
 * a fresh read. NOTE: currently unused (zero call sites in-tree; kept
 * working for API completeness, not exercised by anything today).
 */
void marie_state_inc_tier_seeded(unsigned long pfn, u8 cur);

/*
 * marie_state_move_to_gen - relocate a tracked PFN to @target_gen, with
 * matched marie_track_bm / marie_gen_occupied updates.
 *
 * Single point of policy for any operation that needs to move a folio
 * between gens. Checks marie_state[pfn]'s ISOLATED bit first and declines
 * if set (same rule as marie_state_inc_tier above) -- callers:
 *   - walker promote-on-access (marie_state_inc_tier, this file)
 *   - MADV_COLD (marie_folio_demote, state.c)
 *   - MADV_FREE (lru_marie_lazyfree, state.c)
 *   - Marie defrag's post-migration restamp (defrag.c)
 * Putback (state.c, the release half of the ISOLATED protocol) does NOT
 * go through this helper -- it has its own accounting path since it is
 * also the one clearing ISOLATED; see the ISOLATED gate protocol above.
 *
 * The marie_state[] cmpxchg defeats races against del / another
 * concurrent move. marie_gen_occupied and marie_track_bm are adjusted via
 * marie_bm_set(new plane)/marie_bm_clear(old plane), each gen_occupied
 * update gated on the bitmap call's own atomic 0->1 / 1->0 transition
 * (not unconditionally) -- so a concurrent isolate retiring the same PFN
 * cannot cause a double-count in either direction.
 *
 * No-op if the folio is no longer tracked, is currently ISOLATED, or
 * already encodes @target_gen.
 */
void marie_state_move_to_gen(unsigned long pfn, u8 target_gen);

struct folio;
/*
 * marie_state_drop_pfn - wipe every per-PFN tracking artifact (marie_state
 * byte, marie_track_bm bit, occupancy counter) for @folio. Called from the
 * normal evict path (marie_evict_locked). No-op when the byte is not
 * TRACKED.
 *
 * NOTE: an older revision of this comment additionally described an
 * enable=0 "drain" path (marie_drain_one_lruvec) sharing this helper.
 * Marie's enable/disable is boot-only (lru_marie=0/1, selected before any
 * folio is tracked; see marie_setup() in core.c) -- there is no runtime
 * toggle, and no such drain function exists in this codebase.
 */
void marie_state_drop_pfn(struct folio *folio);

/*
 * marie_state_drop_pfn_isolate - as marie_state_drop_pfn, for the lock-free
 * reclaim context (no lru_lock, IRQs on).
 *
 * Used by the survivor putback's "the isolation ref was the last ref" branch,
 * which must undo the publish it just made. Undoing only the counters there
 * left the byte saying TRACKED && !ISOLATED on a dying folio, which
 * lru_marie_uncharge_backstop() then read as "a credit is outstanding" and
 * debited a second time -- see that branch's comment in state_reclaim.c.
 */
void marie_state_drop_pfn_isolate(struct folio *folio);


/* --- per-folio residency state and install/evict surface --- */
#ifdef CONFIG_LRU_MARIE

#include <linux/atomic.h>
#include <linux/cleanup.h>
#include <linux/gfp_types.h>
#include <linux/hash.h>
#include <linux/irqflags.h>
#include <linux/list.h>
#include <linux/llist.h>
#include <linux/log2.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>

#include <linux/mm_inline.h>
#include <linux/mmzone.h>
#include <linux/percpu.h>
#include <linux/xarray.h>
#include <linux/rcupdate.h>
#include <linux/workqueue.h>
#include <linux/spinlock.h>
#include <linux/swap.h>		/* SWAP_CLUSTER_MAX, ANON_AND_FILE */
#include <linux/types.h>
#include <linux/vmstat.h>

struct folio;
struct lruvec;
struct mem_cgroup;
struct marie_gen;

/*
 * ---------------------------------------------------------------------
 *  Per-folio state inspection (internal)
 * ---------------------------------------------------------------------
 *
 * Reads of the per-PFN state byte are lock-free (READ_ONCE). Writes go
 * through state.c helpers (marie_state_inc_tier, marie_state_move_to_gen,
 * marie_state_drop_pfn); marie_state[] is the single source of truth for
 * Marie's per-folio state.
 *
 * folio->lru is not interpreted as part of Marie's state -- folios
 * are never linked from a Marie-owned list. It exists only so legacy
 * LRU can attach drained folios via lruvec->lists[lru] (handed off by
 * marie_drain_pfn_locked when Marie is disabled).
 */

/**
 * folio_marie_test_tracked - is @folio claimed by Marie?
 *
 * Reads the per-PFN state byte (the single source of truth in the
 * per-PFN paradigm). folio->flags carries no Marie state.
 */
static inline bool folio_marie_test_tracked(const struct folio *folio)
{
	unsigned long pfn = folio_pfn((struct folio *)folio);

	if (!marie_state || pfn >= marie_state_size)
		return false;
	return READ_ONCE(marie_state[pfn]) & MARIE_PFN_TRACKED;
}


/*
 * marie_folio_lruvec_rcu - RCU-bracketed folio_lruvec() for Marie hot paths.
 *
 * folio_lruvec() reaches obj_cgroup_memcg() which has a lockdep predicate
 * requiring rcu_read_lock or cgroup_mutex. Marie's drain and walker paths
 * run with preemption disabled (e.g. under lru_lock) but NOT under
 * rcu_read_lock(); preempt-disable does not satisfy the lockdep
 * predicate. The brief RCU bracket avoids the WARN trip; the returned
 * pointer is used only for equality comparison, never dereferenced
 * after rcu_read_unlock().
 */
static inline struct lruvec *marie_folio_lruvec_rcu(struct folio *folio)
{
	struct lruvec *lv;

	rcu_read_lock();
	lv = folio_lruvec(folio);
	rcu_read_unlock();
	return lv;
}

/*
 * marie_update_lru_size - Marie counterpart to legacy update_lru_size().
 *
 * Maintains two counters for a Marie-tracked folio:
 *
 *   - node NR_LRU_BASE via mod_lruvec_state(), which also folds into the
 *     per-memcg memory.stat breakdown, preserving it for Marie folios;
 *   - the node-global NR_ZONE_LRU_BASE zone counter via
 *     __mod_zone_page_state().
 *
 * Reclaim sizing reads the latter: lru_marie_zone_size_read ->
 * marie_lruvec_zone_size returns the NR_ZONE_LRU_BASE node total
 * directly -- no per-memcg summing, no Marie-private counter.
 *
 * This deliberately does NOT call mem_cgroup_update_lru_size and does NOT
 * write the per-memcg mz->lru_zone_size; that counter is left to
 * legacy/orphan folios only. So Marie<->legacy list transitions stay
 * mz-neutral and there is no stock RMW under lru_lock to underflow.
 *
 * Caller MUST hold lruvec->lru_lock. mod_lruvec_state's per-CPU fold
 * and __mod_zone_page_state's per-zone counter are documented as
 * lru_lock-protected against concurrent updaters of the same lruvec.
 */
static inline void marie_update_lru_size(struct lruvec *lruvec,
				       enum lru_list lru,
				       enum zone_type zid,
				       long nr_pages)
{
	struct pglist_data *pgdat = lruvec_pgdat(lruvec);

	/*
	 * Node NR_LRU_BASE (folds into the per-memcg memory.stat breakdown,
	 * preserving it for Marie folios).
	 */
	mod_lruvec_state(lruvec, NR_LRU_BASE + lru, nr_pages);
	/*
	 * Node-global NR_ZONE_LRU_BASE zone total -- this is what reclaim
	 * sizing reads (marie_lruvec_zone_size). We do NOT touch the per-memcg
	 * mz->lru_zone_size here: it holds only legacy/orphan folios, which
	 * avoids the non-atomic stock RMW under lru_lock and the
	 * "mem_cgroup_update_lru_size: lru_size -N" underflow it used to cause.
	 */
	__mod_zone_page_state(&pgdat->node_zones[zid],
			      NR_ZONE_LRU_BASE + lru, nr_pages);
}

/*
 * Global folio counter, lives in mm/lru_marie/core.c for stats_show;
 * the install/evict helpers in state.c percpu_counter_add it during
 * Marie's TRACKED 0<->1 transitions.
 */
extern struct percpu_counter marie_nr_folios;

/*
 * marie_pc_add - Marie-private percpu_counter add that elides the
 * outer preempt_disable / preempt_enable bracket of
 * percpu_counter_add_batch() while preserving its IRQ safety.
 *
 * percpu_counter_add_batch() wraps the whole body in
 * preempt_disable/enable. Under DEBUG_PREEMPT that bracket shows up in
 * perf under 16-thread memhog as ~4 % of total CPU (preempt_count_add +
 * check_preemption_disabled). We drop it because the individual
 * this_cpu_* primitives used here are each self-contained: this_cpu_add
 * is a single atomic RMW (one instruction on x86), and the slow-path
 * fbc->lock section takes raw_spin_lock_irqsave, so correctness does
 * not depend on the caller's preempt or IRQ state.
 *
 * IRQ safety is MANDATORY, not optional: not every caller holds
 * lru_lock. The reclaim isolate path (marie_evict_counters_only) and
 * the survivor putback in marie_state_shrink_lruvec update the GLOBAL
 * marie_nr_folios counter with IRQs ENABLED (preempt_disable only).
 * The same counter is also bumped from IRQ/softirq context when a
 * Marie-tracked LRU folio's last reference is dropped
 * (folio_put -> __page_cache_release -> lruvec_del_folio ->
 * lru_marie_del_folio -> marie_evict_locked). If the flush path used a
 * plain raw_spin_lock, a softirq landing on the CPU that already holds
 * fbc->lock would spin forever on it -> hard lockup. Hence
 * raw_spin_lock_irqsave below, exactly as percpu_counter_add_batch does.
 *
 * The fast path uses this_cpu_add (atomic against same-CPU IRQ
 * reentrancy); the earlier __this_cpu_read + __this_cpu_write pair was
 * a non-atomic RMW that could lose an IRQ-context update.
 */
static inline void marie_pc_add(struct percpu_counter *fbc, s64 amount)
{
	s64 count = this_cpu_read(*fbc->counters) + amount;

	if (unlikely(abs(count) >= percpu_counter_batch)) {
		unsigned long flags;

		raw_spin_lock_irqsave(&fbc->lock, flags);
		count = __this_cpu_read(*fbc->counters) + amount;
		fbc->count += count;
		__this_cpu_sub(*fbc->counters, count - amount);
		raw_spin_unlock_irqrestore(&fbc->lock, flags);
	} else {
		this_cpu_add(*fbc->counters, amount);
	}
}

/*
 * ---------------------------------------------------------------------
 *  Install / evict — per-folio TRACKED 0 <-> 1 with lru_lock held
 * ---------------------------------------------------------------------
 *
 * Marie's per-folio state is one byte: marie_state[pfn]. Synchronous
 * install/evict helpers own all the bookkeeping (marie_state byte,
 * marie_track_bm bit, global marie_nr_folios percpu_counter, lru_size
 * mirror, PG_active / PG_lru hygiene):
 *
 *   marie_folio_install:  TRACKED 0 -> 1
 *                         unified fresh install for both small folios and
 *                         THP; declared in pfn_install.h
 *   putback (state.c):    reclaim survivor release, TRACKED stays,
 *                         ISOLATED clears, GEN refreshed -- see the
 *                         ISOLATED gate protocol above
 *   marie_evict_locked:   TRACKED 1 -> 0
 *                         called from marie_del_folio_locked
 *
 * folio_marie_test_tracked() is the lock-free state inspector: it
 * reads marie_state[pfn] & MARIE_PFN_TRACKED, returning whether
 * Marie owns @folio. The binary state is checked directly at each
 * callsite -- no intermediate dispatch machinery.
 */
bool marie_evict_locked(struct folio *folio);

/*
 * Reclaim isolate path: sets marie_state[pfn]'s ISOLATED bit at claim
 * time (gating out every other GEN mutator -- walker, MADV_COLD,
 * MADV_FREE, defrag restamp -- for the duration), then decrements
 * gen_occupied for the (gen, type) marie_state[pfn] encodes at that point
 * (read AFTER the ISOLATED CAS, not a value captured earlier during the
 * scan -- see the ISOLATED gate protocol above for why). TRACKED intentionally
 * stays set throughout shrink_folio_list so marie_folio_install's TRACKED
 * early-out blocks any concurrent install from setting PG_lru on a folio
 * currently in the reclaim list. The TRACKED (and ISOLATED) bits are
 * wiped later -- at the buddy free hook (marie_state_drop_pfn_at_free)
 * for a reclaimed folio, or by the putback release path (state.c,
 * clearing ISOLATED last) for a survivor. See state.c body for the full
 * rationale.
 */
void marie_evict_counters_only(struct folio *folio);

/*
 * Canonical per-PFN state teardown invoked from
 * mm/page_alloc.c::free_pages_prepare at every page's buddy handoff.
 * Wipes the per-PFN state byte / bitmap / gen_occupied slot whenever
 * the byte still carries TRACKED. No-op on already-cleared state.
 * Counters are NOT touched (they were balanced upstream by Marie's
 * del path or by marie_evict_counters_only).
 *
 * Lock-free; safe from any context.
 */
void marie_state_drop_pfn_at_free(unsigned long pfn);

/* marie_folio_install lives in pfn_install.h. */

/*
 * Adaptive batch threshold. Returns the per-call page accumulator cap,
 * lerped between MARIE_PFN_BATCH_FLOOR (low pressure,
 * sc->priority == DEF_PRIORITY) and MARIE_PFN_SHRINK_BATCH (max
 * pressure, sc->priority == 0). Defined in state.c as
 * marie_pfn_batch_threshold; this declaration is the public name.
 */
struct scan_control;
unsigned long marie_adaptive_batch_threshold(struct scan_control *sc);

/**
 * marie_del_folio_locked - lru_marie_del_folio body.
 * @folio:            folio to remove (any Marie-tracked state)
 *
 * Universal external-removal handler called from lru_marie_del_folio when
 * lruvec_del_folio fires from outside Marie (compaction, lru_activate
 * batch drain, __page_cache_release after the last folio_put). If the
 * folio is TRACKED, calls marie_evict_locked to run the full eviction
 * (per-PFN state wipe + counter decrements + lru_size mirror). If the
 * folio is no longer TRACKED, returns true defensively (treated as
 * "Marie already removed it").
 *
 * Returns true iff @folio was tracked (the caller can fall through to
 * its remaining bookkeeping). The full counter wind-down -- including
 * the single marie_nr_folios -1 -- is owned by marie_evict_locked via
 * marie_acct_settle_locked (via marie_state_drop_pfn); the caller adds no
 * decrement of its own.
 */
bool marie_del_folio_locked(struct folio *folio);

/*
 * Reclaim-side batch size — fallback compile-time constant used by
 * a few non-hot-path call sites. The per-PFN scan path uses
 * MARIE_PFN_FALLBACK_BATCH / MARIE_PFN_SHRINK_BATCH (see state.c).
 */
#define MARIE_ISOLATE_BATCH SWAP_CLUSTER_MAX

/*
 * Allocation-side aging trigger threshold (per head gen installs). Fully
 * automatic, per-type: marie_gen_growth_threshold(type) (state.c),
 * recomputed at each head advance and on a clean_min_ratio write. Global
 * install cadence: marie_folio_install advances the head gen once the global
 * marie_gen_installs[type] counter reaches marie_gen_growth_threshold(type).
 */

/*
 * ---------------------------------------------------------------------
 *  data structures
 * ---------------------------------------------------------------------
 *
 * Per-type independence is fundamental: anon and file each have their
 * own per-type lock and their own slice of the global per-(type, gen,
 * tier) bitmap / counter arrays. vm.swappiness controls only the
 * eviction proportion between types; aging on one type never forces
 * work on the other.
 *
 * The per-PFN state byte carries the zone field, so per-zone filtering
 * is part of the scan mask -- no per-zone data structure is needed
 * (matching the existing NR_LRU_LISTS / zone semantics).
 */

struct marie_type {
	/*
	 * @type_lock serialises per-type operations that need to be
	 * mutually exclusive across CPUs (THP install, split-tail). Hot
	 * install/del do not take it -- they update the per-PFN state byte
	 * and the unified bitmap lock-free.
	 *
	 * @type: 0 = anon, 1 = file. Set once at marie_type_init time so
	 * scoped_guard(marie_type_lock, ...) can recover the type index
	 * (needed for the per-CPU drain-depth counter) from a bare
	 * struct marie_type * without an extra argument.
	 */
	spinlock_t		type_lock;
	int			type;
};

/*
 * Global per-type locks (one per anon/file).  Marie is desktop/global-only,
 * so the per-type serialising lock is a single global instance, not per
 * lruvec.  Defined in state.c, initialised in marie_counters_init.
 */
extern struct marie_type marie_type_locks[ANON_AND_FILE];

/*
 * Per-type re-entrant-drain detection. Caller (lru_marie_del_folio in
 * mm/lru_marie/core.c) uses marie_in_drain_type(folio's type) to detect "we
 * are already inside a per-type-locked drain for this folio's type on
 * this CPU" and skip the scoped_guard re-acquire. The depth counters
 * are per-CPU statics inside the ADT, mutated by the scoped_guard
 * lock/unlock body (S5 / per-CPU encapsulation).
 */
bool marie_in_drain_type(int type);
void marie_drain_enter_type(int type);
void marie_drain_exit_type(int type);

/*
 * ---------------------------------------------------------------------
 *  drain helpers
 * ---------------------------------------------------------------------
 *
 * No promote-queue or per-CPU staging drain remains: every install /
 * evict / promote-on-access is synchronous (install_local / install_locked
 * publish per-PFN state inline, evict_locked wipes it inline,
 * marie_state_inc_tier promotes via marie_state_move_to_gen directly).
 *
 * No enable/disable drain exists at all: Marie's enable/disable is
 * boot-only (lru_marie=0/1, marie_setup() in core.c), selected before any
 * folio is tracked, so there is nothing to migrate back to the legacy
 * LRU at runtime. (This section's name predates that simplification and
 * the "marie_drain_one_lruvec" it used to describe was already stale
 * before this redesign -- see the note on marie_state_drop_pfn above.)
 */

/*
 * scoped_guard(marie_type_lock, &marie_type_locks[type]) — per-type lock
 * acquisition.
 *
 * Equivalent to the handwritten dance:
 *
 *   spin_lock_irqsave(&t->type_lock, flags);
 *   marie_drain_enter_type(t->type);
 *   ... critical section touching marie_type_locks[t->type] ...
 *   marie_drain_exit_type(t->type);
 *   spin_unlock_irqrestore(&t->type_lock, flags);
 *
 * The cleanup attribute on the guard variable makes the unlock +
 * depth-counter pair a structural property of the scope, not a
 * discipline the caller must remember on every early return / goto.
 *
 * Re-entry inside the scope is handled by the per-CPU per-type
 * marie_drain_depth contract — drain helpers' folio_put recursion that
 * lands in lru_marie_del_folio observes marie_in_drain_type(folio's
 * type) > 0 and skips the spin_lock_irqsave for that type only. Recursion
 * involving the *other* type lands on a depth-0 counter and proceeds to
 * take the corresponding per-type lock as usual (the outer guard holds
 * only one type's lock, so this is not a self-deadlock).
 */
DEFINE_LOCK_GUARD_1(marie_type_lock, struct marie_type,
	/* lock */ ({
		spin_lock_irqsave(&_T->lock->type_lock, _T->flags);
		marie_drain_enter_type(_T->lock->type);
	}),
	/* unlock */ ({
		marie_drain_exit_type(_T->lock->type);
		spin_unlock_irqrestore(&_T->lock->type_lock, _T->flags);
	}),
	unsigned long flags
)

/**
 * marie_counters_init - one-shot init for Marie's global counters.
 *
 * Called from marie_init() (subsys_initcall in mm/lru_marie/core.c).
 * Initialises the global per-type locks (marie_type_locks) and the
 * global marie_nr_folios percpu_counter (the per-CPU bucket pool, slab
 * caches, and cpuhp callbacks that earlier revisions needed have all
 * been retired together with the staging machinery).
 *
 * Returns 0 on success, negative errno on failure (in which case the
 * caller propagates the error up to the initcall machinery).
 */
int marie_counters_init(void);

/*
 * ---------------------------------------------------------------------
 *  Cross-file glue (walker entry points)
 * ---------------------------------------------------------------------
 *
 * These declarations connect mm/lru_marie/core.c (dispatch / lifecycle) and
 * mm/lru_marie/walker.c (PTE walker). They live here to keep mm/
 * private headers down to a single file.
 *
 * Marie holds ZERO per-memcg/per-lruvec state (desktop/global-only): there
 * is no per-lruvec mlv carrier, no lifecycle xarray, no RCU side table.
 */

/*
 * marie_walk_pgdat - run one walker pass for @pgdat.
 *
 * Called from lru_marie_age_node() (kswapd hook) and
 * lru_marie_shrink_lruvec() (direct-reclaim hook). Internally
 * rate-limited per pgdat via a jiffies deadline so calling on every
 * reclaim/kswapd cycle is fine.
 */
void marie_walk_pgdat(struct pglist_data *pgdat);

/*
 * marie_walker_init - one-shot init for the walker subsystem.
 *
 * Initialises per-pgdat bloom-filter spinlocks. Bitmaps themselves
 * are lazily allocated on first Producer hit. Called from
 * marie_init().
 */
void marie_walker_init(void);


#endif /* CONFIG_LRU_MARIE */
#endif /* _MM_LRU_MARIE_STATE_H */
