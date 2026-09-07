/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Hierarchical PFN bitmap backing Marie's global per-(type, zone, gen)
 * planes.
 *
 * Two layers held by one struct:
 *   L1: per-PFN bit, sized BITS_TO_LONGS(max_pfn). set_bit()/clear_bit()
 *       (atomic). One word covers 64 PFNs.
 *
 *   L2: marie_l2_nbits-bit summary, each bit covers (1 << marie_l2_shift)
 *       PFNs (one "L2 range"), sized at boot from marie_max_l2_pages_per_bit
 *       (see below) -- NOT a fixed 512 bits. A companion per-cell
 *       atomic_t refcount tracks how many L1 bits are set in that range.
 *       The L2 bit transitions on the 0 <-> 1 counter boundary, performed
 *       inside the same atomic_*_return path, so concurrent set/clear
 *       cannot desynchronise the bit from the counter.
 *
 * Consumer:
 *   - Global plane: one struct per (type, zone, gen) --
 *     marie_track_bm[type][zone][gen].
 *
 * This is an EXACT structure (unlike the coarse, lossy "hint" plane it
 * replaces): a set L1 bit always means the PFN is genuinely tracked in
 * that exact (type, zone, gen), and the scanner's __ffs-based extraction
 * therefore only ever visits real candidates -- cost is O(occupancy), not
 * O(range). The 2026-08-10 tail /dev/zero incident measured the coarse
 * hint's dense per-PFN walk (marie_state_isolate_scan_l2lock) at 63% of
 * ALL system CPU cycles under low match density; this exact bitmap is the
 * revert back to the pre-redesign design (commit 8f873283e5) that never
 * had that problem, reindexed by zone instead of tier (tier itself is
 * retired -- see state.h) so the scanner can additionally skip whole
 * zone-planes outside sc_reclaim_idx()'s max_zone rather than relying on
 * a per-candidate zone post-filter.
 *
 * marie_max_l2_pages_per_bit (boot param, see marie_bm_global_init): a fixed
 * L2 bit COUNT (the pre-redesign design's original 512) makes pages-per-
 * L2-bit -- and so the number of L1 words the scanner must skim per L2
 * hit before finding actual set bits -- grow linearly with max_pfn. This
 * bounds that skim cost to a constant instead, at the (negligible) cost
 * of more L2 bits -- and so more marie_bm_range_locks entries -- on
 * larger machines. Same idea as the retired marie_hint's
 * marie_hint_pages_per_bit, ported to the exact L1/L2 structure: fix the
 * absolute granularity, let the bit count float with max_pfn, not the
 * other way around.
 *
 * Correctness against the walker/putback race this whole redesign effort
 * exists to close does NOT come from this file: it comes from the
 * MARIE_PFN_ISOLATED gate in marie_state[] (state.h section on the
 * ISOLATED protocol), which every L1-bit mutator (walker inc_tier now
 * "promote to head", MADV_COLD, MADV_FREE, defrag restamp, isolate,
 * putback) must consult before touching a PFN's bit. This file provides
 * only the position/scan-acceleration data structure; ownership is a
 * separate concept, same as the coarse-hint design's split -- only the
 * position structure itself is exact again here.
 *
 * No internal lock; producers serialise via the existing Marie lock
 * hierarchy (lru_lock on the install/del side, marie_bm_range_locks[bit]
 * claim on the scanner side).
 */
#ifndef _MM_LRU_MARIE_BITMAP_H
#define _MM_LRU_MARIE_BITMAP_H

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/bitops.h>
#include <linux/cache.h>
#include <linux/mm.h>		/* max_pfn */
#include <linux/types.h>

/*
 * Maximum PFNs per L2 bit (~32 MiB/bit at 4 KiB pages) -- a CEILING, not
 * a target. Boot-tunable via marie_max_l2_pages_per_bit
 * (lru_marie_max_l2_pages_per_bit= kernel param, core.c). NOT sysfs-tunable:
 * sizes marie_track_bm's L2/range-lock arrays at marie_bm_global_init()
 * (subsys_initcall) time, before any folio is tracked, and resizing them
 * live is not supported.
 *
 * marie_bm_global_init() picks the FINER of two candidate granularities:
 *   - native: max_pfn spread evenly over MARIE_L2_TARGET_BITS (512, "a
 *     handful of cache lines") -- used whenever this is at or under the
 *     ceiling, i.e. the whole plane still fits in ~512 bits.
 *   - ceiling: this constant -- used once max_pfn grows large enough
 *     that native granularity would exceed it. The plane then grows past
 *     MARIE_L2_TARGET_BITS bits (more L2/range-lock cache lines) rather
 *     than letting the per-bit page count grow, since it is bounding
 *     the per-L2-hit L1-word-skim cost (BITS_TO_LONGS(pages/bit)) that a
 *     letting-it-grow design would let scale linearly with max_pfn.
 */
#define MARIE_MAX_L2_PAGES_PER_BIT_DEFAULT	8192

/* Target bit count for one (type, zone, gen) plane -- "a handful of
 * cache lines", not a hard cap; see MARIE_MAX_L2_PAGES_PER_BIT_DEFAULT. */
#define MARIE_L2_TARGET_BITS		512

extern unsigned int marie_max_l2_pages_per_bit;

/*
 * PFN -> L2 bit shift, set at marie_bm_global_init() time so that
 * (1 << marie_l2_shift) PFNs map to one L2 bit -- see
 * MARIE_MAX_L2_PAGES_PER_BIT_DEFAULT for how it is chosen.
 */
extern unsigned int marie_l2_shift;

/* Total L2 bits per (type, zone, gen) plane -- see marie_bm_global_init(). */
extern unsigned int marie_l2_nbits;

static inline unsigned int marie_pfn_to_l2_bit(unsigned long pfn)
{
	unsigned int b = pfn >> marie_l2_shift;

	return b < marie_l2_nbits ? b : marie_l2_nbits - 1;
}

static inline unsigned long marie_l2_bit_pfn_start(unsigned int bit)
{
	return (unsigned long)bit << marie_l2_shift;
}

static inline unsigned long marie_l2_bit_pfn_end(unsigned int bit)
{
	return ((unsigned long)bit + 1) << marie_l2_shift;
}

/*
 * l2/l2_count are separate kvmalloc'd allocations, not inline arrays:
 * marie_l2_nbits is a boot-time value (from marie_max_l2_pages_per_bit), not
 * a compile-time constant, so a fixed-size inline array cannot hold it.
 */
struct marie_bitmap {
	unsigned long	*l1;		/* BITS_TO_LONGS(max_pfn) words */
	unsigned long	*l2;		/* BITS_TO_LONGS(marie_l2_nbits) words */
	atomic_t	*l2_count;	/* marie_l2_nbits entries */
};

/*
 * marie_bm_init - allocate @bm->l1/l2/l2_count. marie_bm_global_init()
 * must have run first (marie_l2_nbits must already be known). Returns 0
 * on success, -ENOMEM on allocation failure.
 */
int marie_bm_init(struct marie_bitmap *bm);

/* marie_bm_free - release @bm's arrays (no-op when never initialised). */
void marie_bm_free(struct marie_bitmap *bm);

/*
 * marie_bm_set - mark @pfn tracked.
 *
 * Atomically sets the L1 bit at @pfn via test_and_set_bit; the per-cell
 * refcount is incremented ONLY on the real 0 -> 1 transition, and on that
 * 0 -> 1 the L2 summary bit for @pfn's range is set. Counting the actual L1
 * transition (rather than incrementing unconditionally) keeps
 * l2_count == popcount(L1 in range) an exact invariant.
 *
 * Returns true iff this call performed the 0 -> 1 transition.
 *
 * static inline because this is a hot-path operation invoked at every
 * install / promote / move; out-of-lining would add a function call +
 * bound-check overhead per call.
 */
static inline bool marie_bm_set(struct marie_bitmap *bm, unsigned long pfn)
{
	unsigned int l2bit;

	if (!bm->l1 || pfn >= max_pfn)
		return false;
	if (test_and_set_bit(pfn, bm->l1))
		return false;		/* already set: keep l2_count == popcount(L1) */
	l2bit = marie_pfn_to_l2_bit(pfn);
	if (atomic_inc_return(&bm->l2_count[l2bit]) == 1)
		set_bit(l2bit, bm->l2);
	return true;			/* this call set the bit (0 -> 1) */
}

/*
 * marie_bm_clear - mark @pfn untracked.
 *
 * Atomically clears the L1 bit at @pfn via test_and_clear_bit; the per-cell
 * refcount is decremented ONLY on the real 1 -> 0 transition, and on that
 * 1 -> 0 the L2 summary bit for @pfn's range is cleared. Decrementing only
 * when this call actually cleared a set bit makes a double-clear harmless
 * instead of underflowing l2_count and desyncing the L2 summary -- the
 * MARIE_PFN_ISOLATED gate (state.h) is what should prevent a double-clear
 * from being attempted in the first place, this is defense in depth.
 */
static inline bool marie_bm_clear(struct marie_bitmap *bm, unsigned long pfn)
{
	unsigned int l2bit;

	if (!bm->l1 || pfn >= max_pfn)
		return false;
	if (!test_and_clear_bit(pfn, bm->l1))
		return false;		/* already clear: do not underflow l2_count */
	l2bit = marie_pfn_to_l2_bit(pfn);
	if (atomic_dec_return(&bm->l2_count[l2bit]) == 0)
		clear_bit(l2bit, bm->l2);
	return true;			/* this call cleared the bit (1 -> 0) */
}

/*
 * marie_bm_test - is @pfn tracked? Lock-free single-word read.
 * Returns false when @bm->l1 is unallocated.
 */
static inline bool marie_bm_test(const struct marie_bitmap *bm,
				 unsigned long pfn)
{
	if (!bm->l1 || pfn >= max_pfn)
		return false;
	return test_bit(pfn, bm->l1);
}

/*
 * marie_bm_drop_l2_range - bulk-clear all L1 / L2 / counter state for the
 * L2 range identified by @l2bit. Used when recycling one range of a
 * bitmap.
 *
 * Caller must guarantee no concurrent set/clear on @bm for the affected
 * PFN range (try_advance_head fences via head_gen cmpxchg).
 */
void marie_bm_drop_l2_range(struct marie_bitmap *bm, unsigned int l2bit);

/*
 * marie_bm_reset - reset @bm to fully empty: L1 cleared, L2 cleared, all
 * l2_count cells zeroed.
 *
 * Used when recycling a (type, zone, gen) slot for the next ring cycle.
 * Caller must fence subsequent installs (head_gen cmpxchg in
 * try_advance_head's case) so no install can target @bm until the reset
 * is visible.
 */
void marie_bm_reset(struct marie_bitmap *bm);

/*
 * L2 range coordination claims: one per L2 bit, used by scanners to claim
 * exclusive ownership of a PFN range for the duration of their L1 walk in
 * that range -- a performance optimisation (avoids duplicate concurrent
 * scanners re-fetching the same L1 cachelines), not a correctness
 * requirement (correctness comes from folio_test_clear_lru() regardless).
 * Shared by ALL marie_bitmap instances: the claim is over the PFN address
 * space, not the bitmap instance.
 *
 * A plain atomic claim flag, NOT spinlock_t. This mirrors the fix in
 * commit e00dbf5a17: nobody ever blocks waiting for one of these (a
 * failed trylock just moves on to the next L2 bit), so there is no
 * "someone else is spin-waiting on me" hazard to justify disabling
 * preemption for the claim's duration. Using spinlock_t here would make
 * cond_resched() a silent no-op inside the claim (spin_trylock() calls
 * preempt_disable() regardless of contention) -- the exact multi-minute
 * scheduler-stall bug already found and fixed once in the coarse-hint
 * design; do not reintroduce it here.
 *
 * SCANNER-ONLY, and it must stay that way. Do not take this around a
 * mutator's marie_state[] update to "keep the byte and the bitmap in sync":
 * that was tried, and it is the wrong shape of fix. The byte is the single
 * source of truth and every counter is derived from ITS transitions
 * (state.h's marie_gen_occ_settle, account.h's marie_acct_settle_*), which
 * makes marie_track_bm a lossy scan index whose disagreement with the byte
 * costs at most a wasted or missed scan pass -- never a counter. There is
 * therefore nothing for a mutator to mutually exclude here, and holding a
 * range claim across a lock-free hot path (promote-on-access) would only add
 * contention to it.
 *
 * Dynamically allocated at marie_bm_global_init() (sized to
 * marie_l2_nbits) rather than a fixed compile-time array, since
 * marie_l2_nbits is a boot-time, not compile-time, value.
 */
struct marie_bm_range_lock {
	atomic_t claimed;	/* 0 = free, 1 = claimed by some scanner */
} ____cacheline_aligned_in_smp;

extern struct marie_bm_range_lock *marie_bm_range_locks;

static inline bool marie_bm_range_trylock(unsigned int l2bit)
{
	return atomic_cmpxchg(&marie_bm_range_locks[l2bit].claimed, 0, 1) == 0;
}

static inline void marie_bm_range_unlock(unsigned int l2bit)
{
	atomic_set(&marie_bm_range_locks[l2bit].claimed, 0);
}

/*
 * marie_bm_global_init - compute marie_l2_shift/marie_l2_nbits from
 * max_pfn and marie_max_l2_pages_per_bit, then allocate marie_bm_range_locks.
 * Must run before any marie_bm_init() call (those need marie_l2_nbits).
 * Returns 0 on success, -ENOMEM on allocation failure.
 */
int marie_bm_global_init(void);

#endif	/* _MM_LRU_MARIE_BITMAP_H */
