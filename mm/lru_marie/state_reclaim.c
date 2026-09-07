// SPDX-License-Identifier: GPL-2.0
/*
 * Marie reclaim engine: the PTE-walker-independent scan/isolate loop,
 * the anon-vs-file swap-bias controller and its supporting pressure
 * heuristics (swap_futile, file_floor_protect, node_under_pressure,
 * file_refaulting), and the shrink_lruvec entry point vmscan actually
 * calls into.
 *
 * This is the vmscan-facing driver layer: it consumes the PFN-state
 * algebra from state_core.c (marie_state[], marie_track_bm[],
 * marie_state_move_to_gen(), marie_try_advance_head_mlv(), ...) and
 * decides what to isolate, when to concede between anon and file, and
 * how much to reclaim per call. Nearly every historical Marie reclaim
 * bug (thrash_wd, FILE_THEN_ANON starvation, swappiness=1 fileonly
 * leaks, aging isolate races) lives in this file.
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


#ifdef CONFIG_X86
#include <asm/cpufeature.h>
#include <asm/processor.h>
#endif

#include "../internal.h"	/* struct scan_control, shrink_folio_list */
#include "state_compat.h"	/* MARIE_FOLIO_FLAGS, marie_shrink_folio_list, marie_account_reclaim */
#include "account.h"
#include "pfn_install.h"
#include "prefetch.h"
#include "state.h"
/*
 * Runtime prefetch-ring parameters, set once at boot by
 * marie_prefetch_params_init() based on CPUID. All values are
 * powers of 2 so the hot path can use & marie_l3_mask instead of
 * % marie_l3_ahead. Defaults are conservative (Silvermont / non-x86).
 */
static unsigned int marie_l3_ahead __read_mostly = 8;
static unsigned int marie_l3_mask  __read_mostly = 7;
static unsigned int marie_l1_ahead __read_mostly = 2;

void __init marie_prefetch_params_init(void)
{
	unsigned int l3 = 8, l1 = 2;

#ifdef CONFIG_X86
	if (!boot_cpu_has(X86_FEATURE_AVX2))
		goto done;

	if (boot_cpu_has(X86_FEATURE_AVX512F)) {
		/* Zen 4/5, Sapphire Rapids: L2 MSHR ~32 */
		l3 = 32; l1 = 8;
		goto done;
	}

	/* AVX2 present but no AVX-512 */
	switch (boot_cpu_data.x86_vendor) {
	case X86_VENDOR_AMD:
		if (boot_cpu_data.x86 >= 0x1A) {
			/* Zen 5+ mobile without AVX-512 */
			l3 = 32; l1 = 8;
		} else if (boot_cpu_data.x86 == 0x19) {
			/* Zen 3 (family 0x19): L2 MSHR ~24 */
			l3 = 24; l1 = 8;
		} else if (boot_cpu_data.x86 == 0x17) {
			/* Zen 1/2 (family 0x17): L2 MSHR ~20 */
			l3 = 20; l1 = 8;
		} else {
			/* Excavator era (family 0x15): L2 MSHR ~12 */
			l3 = 16; l1 = 6;
		}
		break;
	case X86_VENDOR_INTEL:
		/*
		 * CLFLUSHOPT as a Skylake proxy: Haswell and Broadwell
		 * (all models) predate it; Skylake introduced it.
		 */
		if (boot_cpu_has(X86_FEATURE_CLFLUSHOPT)) {
			/* Skylake and newer: L2 MSHR ~20-32 */
			l3 = 24; l1 = 8;
		} else {
			/* Haswell / Broadwell: L2 MSHR ~16 */
			l3 = 16; l1 = 6;
		}
		break;
	default:
		/* Unknown vendor with AVX2: conservative v3 baseline */
		l3 = 16; l1 = 6;
	}
done:
#endif
	marie_l3_ahead = l3;
	marie_l3_mask  = l3 - 1;
	marie_l1_ahead = l1;
	pr_info("prefetch ring: l3_ahead=%u l1_ahead=%u\n", l3, l1);
}

/*
 * Per-CPU shrink scratch buffer, pre-allocated at boot. Reclaim path
 * cannot kmalloc / kvmalloc on the hot path (allocation under memory
 * pressure is what we are trying to relieve), so the isolate batch
 * lives in a fixed per-CPU buffer claimed via an atomic in_use flag.
 * On contention (preempted reclaimer on the same CPU holds the buf
 * across a shrink_folio_list sleep) marie_state_shrink_lruvec falls
 * back to a 160-entry stack array.
 *
 * Sizing: 8192 entries = SWAP_CLUSTER_MAX << 8. Doubled from the
 * MGLRU MAX_LRU_BATCH (4096) reference after boot testing showed
 * 4096-cap reclaim falling behind tail /dev/zero alloc rate. 32 MiB
 * per shrink_folio_list flush at peak amortises lock + IPI overhead
 * twice as well. Per-CPU memory cost:
 *   batch:       8192 * 8 B = 64 KiB
 *   atomic:                = ~4 B
 *   ~= 64 KiB / CPU. 16 CPUs = ~1 MiB system-wide static.
 *
 * Neither PFN nor prev_tier needs its own array at putback: PFN is
 * recovered via folio_pfn(batch[i]), and prev_tier is read back from the
 * per-PFN state byte (counters_only preserves it across isolate).
 */
#define MARIE_PFN_SHRINK_BATCH	(SWAP_CLUSTER_MAX << 8)	/* 8192 */
#define MARIE_PFN_BATCH_FLOOR	(SWAP_CLUSTER_MAX * 8)	/* 256, matches
							 * legacy
							 * MARIE_BATCH_FLOOR */
/*
 * Fallback batch size when the per-CPU buf is contended. 5 *
 * SWAP_CLUSTER_MAX = 160 entries occupy 160 * 8 = 1280 B on the
 * stack; combined with the surrounding ~464 B of non-array locals
 * in shrink_lruvec the frame lands at ~1744 B, staying under the
 * gcc -Wframe-larger-than=2048 threshold without restructuring.
 * 5x SWAP_CLUSTER_MAX.
 */
#define MARIE_PFN_FALLBACK_BATCH (SWAP_CLUSTER_MAX * 5)	/* 160 */

struct marie_shrink_buf {
	atomic_t in_use;
	struct folio *batch[MARIE_PFN_SHRINK_BATCH];
};
static DEFINE_PER_CPU(struct marie_shrink_buf, marie_shrink_buf);

/*
 * Per-PFN adaptive batch threshold, in PAGES (not folios).
 *
 *   priority = DEF_PRIORITY -> floor (MARIE_PFN_BATCH_FLOOR = 256)
 *   priority = 0            -> cap   (MARIE_PFN_SHRINK_BATCH = 8192)
 *
 * This bounds how many pages of isolated-but-not-yet-reclaimed
 * exposure one tier-loop pass may accumulate before calling
 * shrink_folio_list. It must be a PAGE budget, not a folio-count
 * budget: an anon THP is a single folio worth up to HPAGE_PMD_NR
 * (512 on x86-64) pages, so a folio-count cap lets a THP-heavy scan
 * isolate two-plus orders of magnitude more memory than the cap
 * implies. The array capacity (MARIE_PFN_SHRINK_BATCH /
 * MARIE_PFN_FALLBACK_BATCH slots) remains a separate, harder
 * folio-count ceiling enforced at the call site -- this value only
 * gates on n_taken_pages.
 */
static unsigned long marie_pfn_batch_threshold(struct scan_control *sc)
{
	unsigned long floor = MARIE_PFN_BATCH_FLOOR;
	unsigned long cap = MARIE_PFN_SHRINK_BATCH;
	unsigned long pressure;

	pressure = DEF_PRIORITY + 1 -
		   clamp(sc_priority(sc), 0, DEF_PRIORITY);
	return floor + (cap - floor) * (pressure - 1) / DEF_PRIORITY;
}

/*
 * Equivalent of mm/vmscan.c's too_many_isolated() for Marie's own
 * isolate path, which (unlike shrink_inactive_list) never consulted
 * it. Concurrent direct reclaimers isolating from the same node can
 * otherwise pile up unboundedly: each isolated batch is in flight
 * (off the LRU, not yet freed, not counted as free memory) for the
 * duration of shrink_folio_list, which can itself sleep -- e.g. the
 * swap-table cluster allocator's blocking GFP_KERNEL fallback
 * (swap_cluster_alloc_table() in mm/swapfile.c). Without this check
 * that pile-up is unbounded and can drive a zone below its min
 * watermark purely from in-flight isolation, starving even small
 * in-reclaim allocations.
 *
 * kswapd is exempt, matching too_many_isolated(): it is the sole
 * global reclaimer, so throttling it on isolated pages contributed
 * by OTHER (possibly stalled) reclaimers risks stalling the one
 * task relied on to make forward progress.
 */
static bool marie_too_many_isolated(struct pglist_data *pgdat, int type,
				     struct scan_control *sc)
{
	unsigned long inactive, isolated;

	if (current_is_kswapd())
		return false;

	inactive = node_page_state(pgdat, NR_INACTIVE_ANON + type);
	isolated = node_page_state(pgdat, NR_ISOLATED_ANON + type);

	/*
	 * GFP_NOIO/GFP_NOFS callers cannot recurse into the IO/FS paths
	 * that would let them get unstuck, so give them a lower bar to
	 * clear (matches too_many_isolated()'s rationale: avoids a
	 * circular wait against normal, IO-capable reclaimers who ARE
	 * throttled here).
	 */
	if (gfp_has_io_fs(sc_gfp_mask(sc)))
		inactive >>= 3;

	return isolated > inactive;
}

/*
 * marie_state_isolate_scan_l2lock - exact L1/L2-bitmap scan with parallel
 * exclusion via try_lock on per-L2-bit range locks.
 *
 * Loops zone = 0..max_zone; per zone, walks marie_track_bm[type][zone]
 * [oldest_gen]'s L2 summary (512 bits, a handful of cache lines) for ranges
 * that might contain a candidate. For each set L2 bit it try_locks the
 * matching range lock; on success it holds exclusive ownership of that PFN
 * range and __ffs-extracts only the L1 bits actually set in it
 * (O(occupancy), never a dense per-PFN walk). On try_lock failure another
 * scanner already owns the range -- skip and try the next L2 bit. The range
 * lock excludes scanners from each other only; it is a cache-locality
 * optimisation, not a correctness requirement (see bitmap.h), and mutators
 * deliberately do not take it.
 *
 * marie_track_bm is a LOSSY SCAN INDEX over the real state, which is
 * marie_state[]. A PFN whose extracted L1 bit does not match the current
 * marie_state[] byte (wrong type/zone/gen, or ISOLATED) is simply a stale
 * index entry: some mutator moved this PFN's GEN, and the index write that
 * pairs with it either has not landed yet or landed on a different plane.
 * Retire the stale bit and move on. That is safe to do unlocked and
 * best-effort because occupancy is derived from marie_state[]'s transitions
 * (state.h's marie_gen_occ_settle), so clearing an index bit moves no
 * counter -- see the self-heal's own comment in the loop body for what the
 * previous, bitmap-derived scheme made of the same situation.
 *
 * Loop exits when batch_size is reached, nr_to_scan is exhausted, or every
 * L2 bit for every zone in [0, max_zone] has been visited (locked or
 * skipped).
 */
unsigned long marie_state_isolate_scan_l2lock(struct pglist_data *pgdat,
					      int type, int max_zone,
					      struct folio **batch,
					      unsigned long batch_size,
					      unsigned long nr_to_scan,
					      int oldest_in)
{
	u8 oldest_gen, mask, target_base;
	unsigned long start_pfn, end_pfn;
	unsigned long n_batch = 0;
	int zone;

	if (!marie_state)
		return 0;

	/* Gen to scan is the global oldest (already validated >= 0). */
	if (oldest_in < 0)
		return 0;
	oldest_gen = (u8)oldest_in;

	start_pfn = pgdat->node_start_pfn;
	end_pfn   = pgdat_end_pfn(pgdat);
	if (end_pfn > marie_state_size)
		end_pfn = marie_state_size;
	if (start_pfn >= end_pfn)
		return 0;

	if (max_zone < 0)
		return 0;
	if (max_zone >= MARIE_PFN_NR_ZONES_ENCODED)
		max_zone = MARIE_PFN_NR_ZONES_ENCODED - 1;

	mask = MARIE_PFN_TRACKED | MARIE_PFN_TYPE_MASK | MARIE_PFN_ZONE_MASK |
	       MARIE_PFN_GEN_MASK | MARIE_PFN_ISOLATED;
	target_base = MARIE_PFN_TRACKED | (type ? MARIE_PFN_TYPE_FILE : 0) |
		      ((u8)oldest_gen << MARIE_PFN_GEN_SHIFT);

	for (zone = 0; zone <= max_zone && n_batch < batch_size &&
			nr_to_scan > 0; zone++) {
	unsigned long *l1, *l2;
	u8 target;
	unsigned int start_l2, end_l2;
	unsigned int l2_word, l2_word_end;

	{
		struct marie_bitmap *bm =
			&marie_track_bm[type][zone][oldest_gen];

		l1 = bm->l1;
		l2 = bm->l2;
	}
	if (!l1)
		continue;

	target = target_base | marie_pfn_zone_bits(zone);

	start_l2 = marie_pfn_to_l2_bit(start_pfn);
	end_l2 = marie_pfn_to_l2_bit(end_pfn - 1) + 1;
	if (end_l2 > marie_l2_nbits)
		end_l2 = marie_l2_nbits;
	l2_word = start_l2 / BITS_PER_LONG;
	l2_word_end = DIV_ROUND_UP(end_l2, BITS_PER_LONG);

	/*
	 * Outer L2 loop is word-level: the inner __ffs/blsr extraction
	 * visits only set L2 bits of this (type, zone, gen) plane. 512 L2
	 * bits collapse to 8 u64 word iterations; empty words skip at one
	 * cycle each.
	 */
	for (; l2_word < l2_word_end; l2_word++) {
		unsigned long l2w = l2[l2_word];

		/* Mask off pre-start_l2 / post-end_l2 bits in edge words. */
		if (l2_word == start_l2 / BITS_PER_LONG &&
		    (start_l2 % BITS_PER_LONG))
			l2w &= ~((1UL << (start_l2 % BITS_PER_LONG)) - 1);
		if (l2_word + 1 == l2_word_end &&
		    (end_l2 % BITS_PER_LONG))
			l2w &= (1UL << (end_l2 % BITS_PER_LONG)) - 1;

	while (l2w && n_batch < batch_size && nr_to_scan > 0) {
		unsigned int bit = l2_word * BITS_PER_LONG + __ffs(l2w);
		unsigned long lo, hi;
		unsigned long ring[MARIE_L3_AHEAD_MAX];
		int rh = 0, rt = 0, rc = 0;
		unsigned long word_rem;
		unsigned long word_base;
		unsigned long word_i, end_word;
		bool producer_done = false;
		int i, n;
		/* Rate-limits cond_resched() below (defense in depth: this
		 * walk is O(occupancy), not O(range), so it should not run
		 * long enough to matter, but the claim is a plain atomic,
		 * not a spinlock, so yielding here is free -- see bitmap.h). */
		unsigned long visited = 0;
		const unsigned int r_l3_ahead = marie_l3_ahead;
		const unsigned int r_l3_mask  = marie_l3_mask;
		const unsigned int r_l1_ahead = marie_l1_ahead;
		unsigned long state_cl_cursor_l3 = 0;
		unsigned long state_cl_cursor_l1 = 0;
		unsigned long l1_cl_cursor = 0;

		l2w &= l2w - 1;

		if (!marie_bm_range_trylock(bit))
			continue;

		lo = marie_l2_bit_pfn_start(bit);
		hi = marie_l2_bit_pfn_end(bit);
		if (lo < start_pfn)
			lo = start_pfn;
		if (hi > end_pfn)
			hi = end_pfn;

		word_i = lo / BITS_PER_LONG;
		end_word = BITS_TO_LONGS(hi);
		word_base = word_i * BITS_PER_LONG;
		word_rem = (word_i < end_word) ? l1[word_i] : 0;
		if (lo > word_base)
			word_rem &= ~((1UL << (lo - word_base)) - 1);
		word_i++;

#define MARIE_PREFETCH_BMWORD_L3(arr, cursor) do {				\
		unsigned long _bi = word_i + MARIE_BM_L3_AHEAD_WORDS;		\
		if (_bi < end_word) {						\
			unsigned long _cl = (unsigned long)&(arr)[_bi]		\
					    & ~63UL;				\
			if (_cl != (cursor)) {					\
				marie_prefetch_l3((void *)_cl);			\
				(cursor) = _cl;					\
			}							\
		}								\
	} while (0)

#define MARIE_RING_PRODUCE(out_pfn, done_label) do {			\
		while (!word_rem) {					\
			if (word_i >= end_word) {			\
				producer_done = true;			\
				goto done_label;			\
			}						\
			word_rem = l1[word_i];				\
			MARIE_PREFETCH_BMWORD_L3(l1, l1_cl_cursor);	\
			word_base = word_i * BITS_PER_LONG;		\
			word_i++;					\
		}							\
		(out_pfn) = word_base + __ffs(word_rem);		\
		word_rem &= word_rem - 1;				\
		if ((out_pfn) >= hi) {					\
			producer_done = true;				\
			goto done_label;				\
		}							\
	} while (0)

#define MARIE_PREFETCH_STATE_L3(pfn) do {					\
		unsigned long _ah = (pfn) + MARIE_STATE_L3_AHEAD_PFN;		\
		if (_ah < marie_state_size) {					\
			unsigned long _cl = (unsigned long)&marie_state[_ah]	\
					    & ~63UL;				\
			if (_cl != state_cl_cursor_l3) {			\
				marie_prefetch_l3((void *)_cl);			\
				state_cl_cursor_l3 = _cl;			\
			}							\
		}								\
	} while (0)
#define MARIE_PREFETCH_STATE_L1(pfn) do {					\
		unsigned long _ah = (pfn) + MARIE_STATE_L1_AHEAD_PFN;		\
		if (_ah < marie_state_size) {					\
			unsigned long _cl = (unsigned long)&marie_state[_ah]	\
					    & ~63UL;				\
			if (_cl != state_cl_cursor_l1) {			\
				marie_prefetch_l1((void *)_cl);			\
				state_cl_cursor_l1 = _cl;			\
			}							\
		}								\
	} while (0)

		while (rc < r_l3_ahead) {
			unsigned long p;

			MARIE_RING_PRODUCE(p, phase1_done);
			ring[rh] = p;
			rh = (rh + 1) & r_l3_mask;
			rc++;
			MARIE_PREFETCH_STATE_L3(p);
			marie_prefetch_l3(pfn_to_page(p));
		}
phase1_done:

		n = rc < r_l1_ahead ? rc : r_l1_ahead;
		for (i = 0; i < n; i++) {
			unsigned long p = ring[(rt + i) & r_l3_mask];

			MARIE_PREFETCH_STATE_L1(p);
			marie_prefetch_l1(pfn_to_page(p));
		}

		while (rc > 0 && n_batch < batch_size && nr_to_scan > 0) {
			unsigned long pfn = ring[rt];
			u8 s;
			struct folio *f;

			if (!(++visited & 1023))
				cond_resched();

			rt = (rt + 1) & r_l3_mask;
			rc--;
			nr_to_scan--;

			if (!producer_done) {
				unsigned long np;

				MARIE_RING_PRODUCE(np, refill_done);
				ring[rh] = np;
				rh = (rh + 1) & r_l3_mask;
				rc++;
				MARIE_PREFETCH_STATE_L3(np);
				marie_prefetch_l3(pfn_to_page(np));
			}
refill_done:

			if (rc > r_l1_ahead) {
				int idx = (rt + r_l1_ahead - 1) &
					  r_l3_mask;
				unsigned long lp = ring[idx];

				MARIE_PREFETCH_STATE_L1(lp);
				marie_prefetch_l1(pfn_to_page(lp));
			}

			s = READ_ONCE(marie_state[pfn]);
			if ((s & mask) != target) {
				/*
				 * Stale index bit: this plane's bit is set but the
				 * byte -- the single source of truth -- says the
				 * folio is elsewhere, or gone. Retire the bit.
				 *
				 * It moves no counter: gen_occupied is derived from
				 * byte transitions (state.h's marie_gen_occ_settle),
				 * so whoever moved the byte already accounted for it.
				 *
				 * marie_bm_retire, not a bare marie_bm_clear. @s was
				 * read before the clear, so a mutator can move this
				 * PFN INTO this plane in between and have its freshly
				 * published bit removed by us -- leaving the byte
				 * naming a plane with no bit anywhere, which strands
				 * the folio permanently and leaves phantom occupancy
				 * behind it. marie_bm_retire re-reads the byte AFTER
				 * clearing and restores the bit if it turns out to
				 * name this plane; see its comment for why that
				 * converges. It returns true only for a genuinely
				 * stale bit, so the stat stays meaningful.
				 */
				if (marie_bm_retire(pfn, type, zone, oldest_gen))
					atomic_long_inc(&marie_dbg_orphan_bit[type]);
				continue;
			}

			f = pfn_folio(pfn);
			batch[n_batch++] = f;
		}

		marie_bm_range_unlock(bit);
	}	/* while (l2w) -- next set L2 bit in this word */
	}	/* for (l2_word) -- next L2 word */
#undef MARIE_RING_PRODUCE
#undef MARIE_PREFETCH_BMWORD_L3
#undef MARIE_PREFETCH_STATE_L3
#undef MARIE_PREFETCH_STATE_L1
	}	/* for (zone) */

	return n_batch;
}

/*
 * --------------------------------------------------------------------
 *  Anon/file swap-bias controller (stubborn proportional)
 * --------------------------------------------------------------------
 *
 * A single signed counter per marie_lruvec drives the anon-vs-file
 * pick under proportional swappiness (2..199). Granularity rule:
 * EXACTLY ONE type is scanned per shrink_lruvec call in the
 * proportional regime -- the bias sign selects which. Scanning both
 * sides in the same call would dissolve the s:(MAX-s) ratio because
 * every call would contribute pages from both. The caller's priority
 * loop re-enters shrink_lruvec for the next pick, and the bias
 * (updated from this call's outcome) may flip the selection in
 * between -- yielding "fine-grained" type switching at call
 * granularity, which matches the user-visible reclaim cadence.
 *
 *   SUCCESS (nr_reclaimed > 0):
 *     bias += sign * nr_reclaimed * weight
 *     -- page-flow proportional. Long-run pages(anon):pages(file)
 *        converges to s:(MAX_SWAPPINESS-s) even when per-pick batch
 *        sizes differ systematically between types.
 *
 *   FAILURE (nr_reclaimed == 0):
 *     bias unchanged (no-op).
 *     -- The picked side stays the picked side. Failure carries no
 *        back-pressure -- not even a unit nudge -- so the favored
 *        side remains favored indefinitely under sustained failure.
 *        This is the entire point of low-swappiness on modern ZRAM
 *        systems: file should be the eviction target even when it
 *        transiently (or persistently) produces nothing, and anon
 *        must NOT be touched as a consequence of file being stuck on
 *        dirty / locked / writeback / depleted state. If file truly
 *        cannot be reclaimed, the caller escalates priority or OOM
 *        kicks in -- the controller does not surrender protection.
 *
 *   sign = -1 for picked=ANON (push bias toward FILE)
 *          +1 for picked=FILE (push bias toward ANON)
 *   weight = MAX_SWAPPINESS - s   for picked=ANON
 *          = s                    for picked=FILE
 *
 * Special-value swappiness short-circuits the controller:
 *   s=0   FILE only, no fallback (caller proceeds to OOM if depleted)
 *   s=1   FILE first; ANON engages on EITHER of two depletion
 *         signals (see the FILE_THEN_ANON tail gate):
 *           - file < clean_min_ratio floor (skip_file true), or
 *           - file >= floor but the FILE pass FAILED TO MEET this
 *             call's reclaim target = file reclaim is not keeping
 *             pace right now.
 *         Throughput is empirical -- a tracked file folio may be
 *         hot/dirty/mapped, and how much frees is knowable only by
 *         trying -- so the FILE pass's own outcome, not occupancy, is
 *         the signal. Sufficiency (target met) rather than exact-zero
 *         is what keeps reclaim file-first: a positive-but-insufficient
 *         file trickle must not pin reclaim file-only while swappable
 *         anon OOMs with swap free. The fallback fires on the first
 *         call file cannot satisfy -- it does NOT wait for sc->priority
 *         to decay -- and a transient file stall costs at most one
 *         early anon batch; preferred over OOM with swap free.
 *   s=MAX ANON only, no fallback (symmetric to s=0)
 *
 * clean_min_ratio override: when the floor diverts reclaim to
 * anon-only (skip_file in marie_state_shrink_lruvec), the caller
 * does NOT invoke marie_swap_bias_update for that call. The
 * controller stays frozen at its pre-override value so that, when
 * file recovers above the floor, the proportional regime resumes
 * from where it left off -- no post-recovery overshoot from anon
 * reclaim that was driven by external policy, not swappiness.
 *
 * Sysctl writes invoke lru_marie_swappiness_changed() which walks
 * the xarray and resets every swap_bias to zero, so the controller
 * restarts cleanly under the new weight ratio.
 *
 * No CAP: per-cycle delta is bounded by batch_max (~8192) *
 * MAX_SWAPPINESS (200) ~ 1.6e6, far below S64_MAX in any realistic
 * running time. The sysctl-write reset is the only reset mechanism.
 */

/*
 * Swappiness/pick diagnostics (read via /sys/kernel/mm/lru_marie/stats):
 * which pick regime runs, and how many pages Marie's own shrinker reclaims
 * per type. A large reclaimed[anon] vs pswpout means Marie's gate; a small
 * one means the anon comes from elsewhere (legacy orphan drain / non-Marie).
 */
atomic_long_t marie_dbg_pick[5];
atomic_long_t marie_dbg_reclaimed[2];
atomic_long_t marie_dbg_orphan_bit[2];

/* Concede-trigger attribution: [0]=floor [1]=free-pressure [2]=refault [3]=memcg. */
atomic_long_t marie_dbg_concede[4];

/*
 * Second chances the force suppresses: [0]=nr_activate, [1]=nr_ref_keep, per
 * type. Harvested from shrink_folio_list's reclaim_stat -- see state.h.
 */
atomic_long_t marie_dbg_second_chance[2][2];

/*
 * Budget accounting: how much this driver takes against how much the caller
 * asked for. [0]=outstanding need summed at entry, [1]=pages actually
 * delivered, [2]=calls, [3]=largest single-call overshoot (delivered - need).
 *
 * Marie sweeps the whole aged gen ring in ONE call rather than one generation
 * per vmscan priority step, so a single entry can bank far more than the
 * caller's outstanding need even though every individual batch is bounded by
 * marie_pfn_batch_threshold(). This is what those two facts add up to,
 * measured rather than argued.
 */
atomic_long_t marie_dbg_budget[4];

/*
 * Fold one shrink_folio_list pass's second-chance verdicts into the counters.
 * @type indexes Marie's ANON/FILE, matching nr_activate's own indexing.
 */
static void marie_harvest_second_chance(const struct reclaim_stat *stat, int type)
{
	if (type >= 2)
		return;
	if (stat->nr_activate[type])
		atomic_long_add(stat->nr_activate[type],
				&marie_dbg_second_chance[0][type]);
	if (stat->nr_ref_keep)
		atomic_long_add(stat->nr_ref_keep,
				&marie_dbg_second_chance[1][type]);
}

/*
 * GLOBAL anon-vs-file proportional pick bias (desktop/global-only). Signed:
 * < 0 favours FILE, >= 0 favours ANON. A single node-wide controller drives
 * every reclaim pass.
 */
atomic64_t marie_swap_bias;

enum marie_pick_kind marie_swap_pick_type(u8 swappiness)
{
	if (swappiness == 0)
		return MARIE_PICK_FILE_STRICT;
	if (swappiness == 1)
		return MARIE_PICK_FILE_THEN_ANON;
	if (swappiness >= MAX_SWAPPINESS)
		return MARIE_PICK_ANON_STRICT;

	return (atomic64_read(&marie_swap_bias) < 0)
		? MARIE_PICK_FILE_FIRST
		: MARIE_PICK_ANON_FIRST;
}

void marie_swap_bias_update(int picked_type,
			    unsigned long nr_reclaimed,
			    u8 swappiness)
{
	s64 delta;

	/*
	 * Special values bypass the controller. The pick path does not
	 * read swap_bias under {0, 1, MAX_SWAPPINESS}, so the value
	 * here is irrelevant to observable behaviour; skipping the
	 * write also avoids gratuitous cache-line bouncing.
	 */
	if (swappiness <= 1 || swappiness >= MAX_SWAPPINESS)
		return;

	/*
	 * Failure carries no back-pressure: when nr_reclaimed is zero,
	 * the bias is left untouched. The picked side stays the picked
	 * side -- truly stubborn protection of the favored type. See
	 * the top of this section for the failsafe semantics.
	 */
	if (!nr_reclaimed)
		return;

	if (picked_type == 0)
		delta = -(s64)nr_reclaimed *
			(s64)(MAX_SWAPPINESS - swappiness);
	else
		delta = +(s64)nr_reclaimed * (s64)swappiness;

	atomic64_add(delta, &marie_swap_bias);
}

/*
 * marie_file_floor_protect - is the clean_min_ratio file floor in force?
 *
 * Returns true when this node's clean file pagecache has fallen TO OR BELOW
 * marie_clean_min_ratio (% of node_present_pages) and Marie still has
 * anon to absorb the pressure, so file reclaim must be withheld. The pick
 * driver diverts file -> anon on this signal (skip_file) and folds the
 * result into the MARIE_DRAIN_* mask it returns, so shrink_lruvec's legacy
 * orphan drain spares file too. No reclaim path may evict file below the
 * floor -- le9uo's single-path floor invariant applied across Marie's paths.
 *
 * Only CLEAN file counts toward the floor (NR_FILE_DIRTY subtracted):
 * dirty pages cannot be reclaimed without writeback, so counting them
 * would let the floor be satisfied by unreclaimable pages and strand the
 * clean working set.
 *
 * If anon is empty Marie has no reserve to protect anyway, so the floor
 * yields and file scan proceeds as a last resort. An OOM victim bypasses
 * the floor entirely (its file is fair game; see the oom_victim handling
 * in marie_state_shrink_lruvec).
 */

static bool marie_file_floor_protect(struct pglist_data *pgdat)
{
	unsigned int min_ratio = READ_ONCE(marie_clean_min_ratio);
	unsigned long file_pages, dirty, file_min;
	long anon_occupied = 0;
	int g;

	if (!min_ratio || unlikely(tsk_is_oom_victim(current)))
		return false;

	file_pages = node_page_state(pgdat, NR_ACTIVE_FILE) +
		     node_page_state(pgdat, NR_INACTIVE_FILE);
	dirty = node_page_state(pgdat, NR_FILE_DIRTY);
	file_pages = (file_pages > dirty) ? file_pages - dirty : 0;
	file_min = pgdat->node_present_pages * min_ratio / 100;

	/*
	 * Strict '>' (not '>='): clean file AT EXACTLY the floor must be
	 * protected, not reclaimed. Under swappiness=1 the FILE_THEN_ANON tail
	 * concedes to anon iff this returns true (concede == floor_protect for
	 * global reclaim); with '>=', file held at the floor by pagecache refill
	 * never crosses STRICTLY below it, so concede never fires and GBs of cold
	 * anon are stranded behind a file refill treadmill -- the allocator
	 * livelocks (anon still counts as reclaimable, so the OOM gate never
	 * trips) instead of swapping. Conceding at the floor is the intended
	 * "file first, until the floor, then anon".
	 */
	if (file_pages > file_min)
		return false;

	for (g = 0; g < MARIE_PFN_NR_GENS; g++)
		anon_occupied += atomic_long_read(&marie_gen_occupied[g][0]);

	return anon_occupied > 0;
}

/*
 * marie_node_under_pressure - is this node failing to hold free above its
 * watermarks?
 *
 * The FILE_THEN_ANON tail (swappiness=1, Marie's default) concedes to anon
 * only once clean file has drained BELOW the clean_min_ratio floor. That DEFER
 * assumes successive file-only calls DRAIN clean file to the floor. A workload
 * that refills clean file above the floor faster than the batch-capped sweep
 * drains it -- a compile streaming its object cache, dozens of mmap'd binaries
 * whose text re-faults the moment it is dropped -- holds the floor forever
 * unreached: the aged FILE ring keeps depleting (fresh reads land in the young
 * head gen), the FILE pass falls short every time it drains it, yet the DEFER
 * fires because the total file LEVEL is still above the floor. Free stays
 * pinned at the watermarks while GBs of swappable anon -- including swap-backed
 * shmem (tmpfs, memfd/ZGC heaps) -- are never offered to swap: the box thrashes
 * on the file refill treadmill and OOMs with swap free.
 *
 * When free has actually fallen to the watermarks the file level is moot: the
 * allocator cannot build headroom, so offer anon to swap now regardless of the
 * floor. Same free <= 2*high gate the thrash watchdog uses
 * (thrash_wd_mem_pressured in oom_kill.c), applied here one layer earlier so
 * anon reaches swap as pressure builds rather than only after the watchdog's
 * multi-second stall. A busy-but-healthy large-RAM box streaming page cache
 * keeps free well above the watermarks (the FILE pass also meets its target
 * from the abundant cache and never reaches this tail), so this stays false
 * and file-first is preserved.
 */
static bool marie_node_under_pressure(struct pglist_data *pgdat)
{
	unsigned long free = 0, high = 0;
	int z;

	for (z = 0; z < MAX_NR_ZONES; z++) {
		struct zone *zone = &pgdat->node_zones[z];

		if (!managed_zone(zone))
			continue;
		free += zone_page_state(zone, NR_FREE_PAGES);
		high += high_wmark_pages(zone);
	}

	return free <= high * 2;
}

/*
 * marie_file_refaulting - is reclaim evicting FILE that comes straight back?
 *
 * The refault-feedback pressure signal (concede_pressure_mode & REFAULT).
 * WORKINGSET_REFAULT_FILE counts file pages faulted back in that carried an
 * eviction shadow -- pages we dropped and that returned. When that runs at
 * >= half of Marie's own file reclaim rate over the sample window, the
 * "clean" file we keep dropping IS the working set (hot; a refill/refault
 * treadmill), so file-first is futile -- concede to anon regardless of the
 * clean_min_ratio floor or the free level. Unlike free<=2*high this measures
 * the pathology (are the evicted pages actually cold?) directly, and fires as
 * file goes hot rather than only after free has cratered.
 *
 * Sampled globally (Marie is desktop/global-only) at most ~4x/s under a
 * trylock; concurrent reclaimers read the cached boolean. First refresh only
 * seeds the baseline (no bogus cumulative delta). The 2x weight mirrors the
 * thrash watchdog's refault:steal ratio in mm/oom_kill.c.
 */
static DEFINE_SPINLOCK(marie_rf_lock);
static unsigned long marie_rf_next;			/* jiffies of next refresh */
static unsigned long marie_rf_last_rf, marie_rf_last_st;	/* file refault / reclaim */
static bool marie_rf_primed;
static bool marie_rf_file_hot;

static bool marie_file_refaulting(void)
{
	unsigned long now = jiffies, rf, st, drf, dst;

	if (time_before(now, READ_ONCE(marie_rf_next)))
		return READ_ONCE(marie_rf_file_hot);
	if (!spin_trylock(&marie_rf_lock))
		return READ_ONCE(marie_rf_file_hot);
	if (time_before(now, marie_rf_next)) {		/* lost the refresh race */
		spin_unlock(&marie_rf_lock);
		return READ_ONCE(marie_rf_file_hot);
	}

	rf = global_node_page_state(WORKINGSET_REFAULT_FILE);
	st = atomic_long_read(&marie_dbg_reclaimed[1]);		/* type 1 == file */
	WRITE_ONCE(marie_rf_next, now + HZ / 4);

	if (unlikely(!marie_rf_primed)) {
		marie_rf_last_rf = rf;
		marie_rf_last_st = st;
		marie_rf_primed = true;
		spin_unlock(&marie_rf_lock);
		return false;
	}

	drf = rf - marie_rf_last_rf;
	dst = st - marie_rf_last_st;
	marie_rf_last_rf = rf;
	marie_rf_last_st = st;
	/* >= half of the file reclaimed this window faulted straight back. */
	WRITE_ONCE(marie_rf_file_hot, dst && drf * 2 >= dst);
	spin_unlock(&marie_rf_lock);

	return READ_ONCE(marie_rf_file_hot);
}

/*
 * marie_state_shrink_lruvec - per-PFN paradigm reclaim driver.
 *
 * Aging has two head-advance triggers. SUPPLY-PUSH: install cadence
 * advances the head every marie_gen_growth_live[type] installs
 * (marie_folio_install). DEMAND-PULL: when a type's sweep finds the aged
 * ring exhausted (find_oldest < 0) but cold pages are parked in the head
 * gen, it seals the head so they age into reclaim range -- the
 * reclaim-driven trigger, fired ONLY on true exhaustion (the retired
 * unconditional "occupied < 2 at entry" form thrashed the ring). Without
 * demand-pull a workload that stops installing a type strands its
 * head-gen pages, since marie_find_oldest_occupied skips head (the
 * install destination).
 *
 * Per (type, tier) the scan walks the per-(gen, type) bitmap, claims
 * each candidate via folio_try_get + folio_test_clear_lru, then calls
 * marie_evict_counters_only: counters decremented and the scan-bitmap
 * slot + gen_occupied retired at isolate (so other CPUs stop re-finding
 * the in-flight folio), but the per-PFN TRACKED byte is KEPT so
 * install_local's early-out blocks a concurrent install from re-setting
 * PG_lru while shrink_folio_list reclaims it.
 *
 * Teardown of the TRACKED byte is deferred: a reclaimed folio is wiped
 * at its buddy handoff (marie_state_drop_pfn_at_free via the
 * free_pages_prepare hook), which finds the scan bit already clear and
 * so does not double-decrement l2_count / gen_occupied. Survivors of
 * shrink_folio_list keep TRACKED and are re-published at the putback gen
 * via marie_state_move_to_gen (set-new + clear-old, gated individually on
 * each bit's own transition -- NOT a publish-only set: the walker's
 * marie_state_inc_tier races on the state byte alone, with no scan-bitmap
 * or PG_lru gate, so it can relocate this exact PFN while it sits isolated
 * in shrink_folio_list; move_to_gen clears whatever slot the byte
 * currently points to, wherever the walker last left it, instead of
 * assuming isolate's original slot is the only one to retire), seeding
 * tier from max(prev_tier, PG_active/PG_workingset).
 */


unsigned int marie_state_shrink_lruvec(struct lruvec *lruvec, struct scan_control *sc)
{
	struct pglist_data *pgdat = lruvec_pgdat(lruvec);
	struct mem_cgroup *memcg = lruvec_memcg(lruvec);
	/*
	 * Budget accounting (see marie_dbg_budget). The caller's target covers
	 * the whole node walk and sc->nr_reclaimed already holds what earlier
	 * lruvecs banked, so the need THIS entry is responsible for is the
	 * difference -- comparing against nr_to_reclaim alone would understate
	 * the overshoot on every call after the first.
	 */
	unsigned long budget_target = sc_nr_to_reclaim(sc);
	unsigned long budget_done = sc_nr_reclaimed(sc);
	unsigned long budget_need = budget_target > budget_done ?
				    budget_target - budget_done : 0;
	/*
	 * Desktop/global-only Marie: the scan is ALWAYS global -- there is no
	 * per-memcg reclaim, so even a cgroup-targeted shrink scans the whole
	 * node's oldest gen (best-effort; memory.max is not enforced via
	 * reclaim).
	 */
	/*
	 * @swappiness is captured once per call; subsequent sysctl
	 * writes that reset the global bias to zero are seen on the NEXT
	 * call. mem_cgroup_swappiness returns the effective value (memcg own
	 * value on cgroup v1 non-root, vm_swappiness otherwise) and is
	 * a plain READ_ONCE under the hood.
	 *
	 * low_swappiness_mode (default on) clamps the effective value to at
	 * most 1 (MARIE_PICK_FILE_THEN_ANON), Marie's recommended policy,
	 * regardless of the higher values vm.swappiness / memory.swappiness
	 * udev rules, tuning daemons, or distro defaults have installed. It
	 * only ever LOWERS swappiness, so the special "never swap" value 0
	 * (MARIE_PICK_FILE_STRICT: OOM rather than touch anon) is preserved --
	 * an operator who deliberately set 0 still gets 0. Clear the knob to
	 * honour the configured value verbatim. See the rationale at the top
	 * of core.c.
	 */
	u8 configured = (u8)mem_cgroup_swappiness(memcg);
	u8 swappiness = (READ_ONCE(marie_low_swappiness_mode) && configured > 1) ?
			1 : configured;
	enum marie_pick_kind pick_kind;
	int type_order[2];
	int type_count;
	int idx;
	bool skip_file = false;
	unsigned int drain_mask;
	/*
	 * When anon cannot be reclaimed at all (no free swap slots,
	 * cgroup swap limit hit, no demotion target), swappiness is by
	 * definition meaningless -- it expresses the anon:file reclaim
	 * ratio, and one side of that ratio no longer exists. Every ANON
	 * pick would reclaim nothing, and because the bias controller
	 * takes no back-pressure from a zero-reclaim pick
	 * (marie_swap_bias_update bails on !nr_reclaimed), the bias never
	 * flips to FILE: reclaimable file cache is stranded until OOM.
	 * Drop the stubborn swappiness preference and force FILE only,
	 * mirroring get_scan_count()'s "!can_reclaim_anon_pages ->
	 * SCAN_FILE". The clean_min_ratio floor below still applies, so
	 * file is reclaimed only down to the protected floor; once file is
	 * at the floor and anon is unreclaimable this pass reclaims nothing,
	 * and the stock no_progress_loops path in should_reclaim_retry()
	 * reaches the OOM killer.
	 *
	 * The RAM-backed-swap case -- where anon keeps counting as
	 * reclaimable because swap slots are nominally plentiful, while each
	 * swap-out consumes almost as much RAM as it frees -- needs no gate
	 * here. zone_reclaimable_pages() discounts anon by the compressed
	 * store's own footprint (mm/vmscan.c, gated on lru_marie_enabled()),
	 * so should_reclaim_retry()'s existing watermark arithmetic reaches
	 * the OOM path on its own. Nothing is
	 * on the existing FILE_STRICT -> stock-retry-exhaustion path: Marie
	 * never calls out_of_memory() itself.
	 */
	bool anon_unreclaimable =
		!vmscan_can_reclaim_anon_pages(memcg, pgdat->node_id, sc);
	/*
	 * An OOM victim's own direct reclaim runs FILE-only, with no holds
	 * barred on the file side: scan FILE ignoring the swappiness/bias
	 * pick, the clean_min_ratio floor, the FILE_THEN_ANON tail gate and
	 * the bias controller. The task has been selected for death and the
	 * OOM reaper frees its anon, so swapping anon here would only add
	 * I/O thrash for no benefit -- reclaim just the cheap, no-I/O file
	 * side (clean_min_ratio is bypassed below, so all file is fair
	 * game). If file is exhausted the victim falls back on the reaper,
	 * which is the normal OOM mechanism. kswapd is never an OOM victim,
	 * so background reclaim is unaffected.
	 */
	bool oom_victim = tsk_is_oom_victim(current);
	int type;

	/*
	 * No head advance here. Aging is driven by install cadence
	 * (marie_folio_install advances the head every
	 * marie_gen_growth_live[type] installs, under lru_lock). The old
	 * reclaim-time "occupied < 2" advance thrashed the ring under
	 * concurrent reclaim; see marie_try_advance_head_mlv.
	 */

	/*
	 * clean_min_ratio hard floor. True when this node's clean file
	 * pagecache is below the configured percentage of node_present_pages
	 * (and anon remains, and we are not an OOM victim). The same predicate
	 * masks the legacy drain's file scan in shrink_lruvec, so no path
	 * evicts file below the floor (le9uo's single-path floor invariant).
	 */
	/*
	 * Resample the install-cadence base here as well as at head advance.
	 * Its free-memory term is exactly what moves under pressure, and an
	 * advance-only refresh would hold one sample across a whole generation's
	 * worth of installs -- longest precisely when the head is advancing
	 * slowly, which is when it matters most. Reclaim entry is off the install
	 * path and only runs when there is pressure to track.
	 */
	marie_recompute_growth_base(0);
	marie_recompute_growth_base(1);

	skip_file = marie_file_floor_protect(pgdat);

	/*
	 * Choose the type(s) to scan as a strict priority cascade:
	 *
	 *   oom_victim         -> FILE only. The victim's anon is reaped by the
	 *                         OOM reaper, so swapping anon is pure I/O thrash;
	 *                         reclaim the cheap no-I/O file side. The floor is
	 *                         bypassed for victims (skip_file is false), so
	 *                         file scans freely.
	 *   anon_unreclaimable -> FILE only. No free swap slots / no demotion
	 *                         target: swappiness is meaningless and every ANON
	 *                         pick would free nothing. If file is also at the
	 *                         floor the per-iteration gate no-ops the file
	 *                         scan and the stock no_progress_loops path OOMs.
	 *   swappiness == 0     -> FILE only. Hard "never swap" user policy: the
	 *                         clean_min_ratio floor must NOT punch through it
	 *                         (core.c). At the floor file is blocked too, so
	 *                         this OOMs rather than swapping -- the contract.
	 *   skip_file          -> ANON only. The floor is in force and file is
	 *                         protected, so divert all reclaim to anon
	 *                         regardless of the swappiness/bias pick. This
	 *                         outranks the proportional controller: a
	 *                         FILE_FIRST pick would otherwise scan the
	 *                         floor-blocked file side, free nothing, and --
	 *                         the bias being frozen during skip_file -- stay
	 *                         pinned on FILE while anon is never picked,
	 *                         stalling reclaim under pressure at high swappiness.
	 *   otherwise          -> the swappiness / swap_bias proportional pick.
	 */
	if (oom_victim)
		pick_kind = MARIE_PICK_FILE_STRICT;
	else if (anon_unreclaimable)
		pick_kind = MARIE_PICK_FILE_STRICT;
	else if (swappiness == 0)
		pick_kind = MARIE_PICK_FILE_STRICT;
	else if (skip_file)
		pick_kind = MARIE_PICK_ANON_STRICT;
	else
		pick_kind = marie_swap_pick_type(swappiness);

	if ((unsigned int)pick_kind < ARRAY_SIZE(marie_dbg_pick))
		atomic_long_inc(&marie_dbg_pick[pick_kind]);

	switch (pick_kind) {
	case MARIE_PICK_FILE_STRICT:
		type_order[0] = 1;
		type_count = 1;
		break;
	case MARIE_PICK_ANON_STRICT:
		type_order[0] = 0;
		type_count = 1;
		break;
	case MARIE_PICK_FILE_THEN_ANON:
		/*
		 * swappiness=1: FILE first, ANON as the depletion fallback
		 * the moment FILE fails to satisfy this call's reclaim
		 * target (not only when FILE returns exactly zero).
		 * type_count=2 with the sufficiency gate at the tail.
		 */
		type_order[0] = 1;
		type_order[1] = 0;
		type_count = 2;
		break;
	case MARIE_PICK_FILE_FIRST:
		/*
		 * Proportional regime, bias picks FILE. SINGLE type per
		 * call: scanning the other side in the same call would
		 * dissolve the s:(MAX-s) ratio because both sides would
		 * contribute pages on every invocation. The caller
		 * (vmscan priority loop) re-enters shrink_lruvec for
		 * the next pick; bias may flip in between via the
		 * proportional update from this call's outcome.
		 */
		type_order[0] = 1;
		type_count = 1;
		break;
	case MARIE_PICK_ANON_FIRST:
	default:
		/* Symmetric: proportional regime, bias picks ANON. */
		type_order[0] = 0;
		type_count = 1;
		break;
	}

	/*
	 * Tell shrink_lruvec which orphan type(s) its legacy drain may
	 * reclaim: exactly the type this call scans. type_order[0] is the
	 * primary (and, in the single-type regime, only) type. A file pick
	 * blocked by skip_file (FILE_STRICT under the clean_min_ratio floor)
	 * scans nothing, so it grants no drain -- preserving the
	 * no-progress -> OOM path.
	 */
	if (type_order[0] == 1)
		drain_mask = skip_file ? 0 : MARIE_DRAIN_FILE;
	else
		drain_mask = MARIE_DRAIN_ANON;

	{
		/*
		 * Claim this CPU's pre-allocated shrink buffer. If the
		 * cmpxchg fails (preempted reclaimer on the same CPU
		 * holds it across a shrink_folio_list sleep), fall back
		 * to a small stack batch.
		 */
		struct marie_shrink_buf *buf;
		/*
		 * Fallback uses MARIE_PFN_FALLBACK_BATCH-sized stack
		 * arrays. Sized to stay under gcc -Wframe-larger-than=2048
		 * given the ~464 B baseline frame; see MARIE_PFN_FALLBACK_
		 * BATCH comment.
		 */
		struct folio *small_batch[MARIE_PFN_FALLBACK_BATCH];
		struct folio **scratch_batch;
		/* Hard folio-count ceiling: the scratch array's actual slot
		 * count. Never exceeded regardless of folio order. */
		unsigned long array_cap;
		/* Soft PAGE-count budget: see marie_pfn_batch_threshold(). */
		unsigned long batch_max;
		bool using_percpu;

		buf = per_cpu_ptr(&marie_shrink_buf, raw_smp_processor_id());
		if (atomic_cmpxchg(&buf->in_use, 0, 1) == 0) {
			scratch_batch = buf->batch;
			array_cap = MARIE_PFN_SHRINK_BATCH;
			batch_max = marie_pfn_batch_threshold(sc);
			using_percpu = true;
		} else {
			scratch_batch = small_batch;
			array_cap = MARIE_PFN_FALLBACK_BATCH;
			batch_max = MARIE_PFN_FALLBACK_BATCH;
			using_percpu = false;
		}

		for (idx = 0; idx < type_count; idx++) {
			int oldest;
			LIST_HEAD(folio_list);
			struct reclaim_stat stat = {};
			unsigned long n_taken = 0;
			/* PAGE count of isolated folios (n_taken counts folios; a
			 * THP is folio_nr_pages pages). NR_ISOLATED_* and PGSCAN_*
			 * are page counters by kernel convention, so they must use
			 * this, not n_taken -- else THP undercounts NR_ISOLATED and
			 * too_many_isolated() under-throttles concurrent reclaim. */
			unsigned long n_taken_pages = 0;
			unsigned int n_reclaimed = 0;
			int oldest_for_putback;
			u8 putback_gen;
			struct folio *f, *tmp;
			/* Per-sweep reclaim accumulator across the aged gen ring. */
			unsigned long total_reclaimed = 0;
			int sweep_i;
			/*
			 * Tracks whether this iteration actually attempted
			 * to pick the type. An external override
			 * (skip_file from clean_min_ratio) clears this so
			 * the bias controller is NOT updated for a pick
			 * that never ran -- the bias must reflect actual
			 * picking policy, not blocked intentions.
			 */
			bool attempted_pick = true;

			type = type_order[idx];

			/*
			 * SWEEP the aged gen ring (oldest -> head) for this type
			 * in ONE call: reclaim until the target is met (goto
			 * done) or every aged gen has been visited and reclaim
			 * still fell short. Bounded by MARIE_PFN_NR_GENS
			 * iterations -- there are only NR_GENS-1 non-head gens,
			 * and survivors re-publish ahead of the oldest pointer
			 * (hot -> head, cold -> oldest+1) so find_oldest advances
			 * monotonically toward head. This makes "swept the whole
			 * ring and still short" a direct, content-based failure
			 * signal for the s1 gate below, replacing the former
			 * one-gen-per-call scan that leaned on vmscan's priority
			 * loop (and decayed priority) to drive the sweep.
			 *
			 * A plain `break` (skip_file, ring empty, nothing
			 * isolatable) drops to the per-type tail; `goto done`
			 * (target reached) bypasses the tail entirely.
			 */
			for (sweep_i = 0; sweep_i < MARIE_PFN_NR_GENS; sweep_i++) {

			if (type == 1 && skip_file) {
				attempted_pick = false;
				break;
			}

			/* Reset per-gen scratch for this sweep step. */
			INIT_LIST_HEAD(&folio_list);
			stat = (struct reclaim_stat){};
			n_taken = 0;
			n_taken_pages = 0;
			n_reclaimed = 0;

			oldest = marie_find_oldest_occupied_mlv(type);
			if (oldest < 0) {
				u8 cur_head;
				/*
				 * Demand-pull aging. The aged ring is out of
				 * reclaimable pages of this type, but cold/clean
				 * pages may be PARKED in the head gen: supply-push
				 * (install cadence) advances the head only as new
				 * pages of this type are installed, so a workload
				 * that stops installing it (a static file cache; a
				 * cold anon burst touched once then idle) strands
				 * its head-gen pages, which find_oldest -- skipping
				 * the head by design -- cannot reach. Seal the head
				 * so they age into reclaim range, then retry the
				 * sweep.
				 *
				 * This is the reclaim-driven counterpart to
				 * supply-push -- the retired occupied<2 trigger,
				 * restored in demand-pull form: it fires ONLY on
				 * true aged exhaustion (find_oldest<0), not at every
				 * shrink entry (the unconditional firing is what
				 * thrashed the ring), and is bounded by the
				 * MARIE_PFN_NR_GENS sweep cap. Needed for ANON too:
				 * without it a cold anon burst parked at the head
				 * could never be swapped under high swappiness. For
				 * FILE the clean_min_ratio floor gates it so the
				 * reserve is not breached; anon has no such reserve.
				 */
				bool may_advance = (type != 1) ||
					!marie_file_floor_protect(pgdat);

				if (may_advance) {
					cur_head = (u8)atomic_read(
						&marie_head_gen[type]);
					/*
					 * Only seal a NON-EMPTY head. Advancing an
					 * empty head would recycle it into a sealed
					 * empty gen -- a hole that find_oldest skips
					 * and that nothing fills until the head laps
					 * the ring. The other advance site (install
					 * cadence in marie_folio_install) is likewise
					 * non-empty: it fires right after the publish
					 * that bumped gen_occupied[head]. So no
					 * caller of marie_try_advance_head_mlv ever
					 * advances an empty head.
					 */
					if (atomic_long_read(
						&marie_gen_occupied[cur_head][type]) > 0 &&
					    marie_try_advance_head_mlv(type))
						continue;
				}
				break;
			}
			/*
			 * References are ALWAYS consulted here. There is no
			 * force-reclaim override any more; @ignore_refs stays false
			 * and shrink_folio_list runs folio_check_references() on
			 * every folio Marie isolates.
			 *
			 * What used to be here was a gate meant to certify that
			 * aging had already looked at this generation, and so that
			 * the reference bits left on its folios were stale:
			 *
			 *   ignore_refs = (aging_epoch[type] -
			 *                  recycle_epoch[oldest][type]) > 0;
			 *
			 * It certified nothing. The two clocks it compared run at
			 * rates two to three orders of magnitude apart --
			 * marie_aging_epoch ticked once per walker pass (32..1000
			 * ms) while marie_recycle_epoch was stamped once per head
			 * advance, measured at one per ~192 s for FILE on a live
			 * 30 GiB desktop -- so a strict `> 0` was satisfied by a
			 * single tick and stayed satisfied until the next advance.
			 * Measured on that desktop: 663760 FILE batches forced,
			 * 0 respected. Not 99%. All of them. A latch, not a gate,
			 * which is the same objection that had already retired the
			 * swappiness policy term sharing this flag -- and the
			 * measurement offered to justify THAT removal (13271 of
			 * 13271 batches) was in hindsight evidence that this term
			 * was equally degenerate, not that the other was redundant.
			 * Removing one always-true condition left the other one
			 * always true, so the VM_EXEC suppression that removal was
			 * meant to end simply continued.
			 *
			 * Three things settled it, none of them performance:
			 *
			 *   1. folio_referenced() returns before rmap_walk() when
			 *      folio_mapcount() is zero, so the ~1 us/folio the gate
			 *      was introduced to save (a claimed 5% CPU win) never
			 *      existed for unmapped page cache -- the one population
			 *      where forcing is sound. It was real only for mapped
			 *      folios, where forcing is not.
			 *   2. MGLRU answers the identical "two second-chance
			 *      mechanisms in series" question with a multi-gen ring
			 *      of its own by passing ignore_references=false
			 *      unconditionally (evict_folios). Across the tree only
			 *      reclaim_folio_list() passes true, and there it means
			 *      a command -- the caller named these pages -- not an
			 *      aging policy. Marie was the only policy user.
			 *   3. folio_check_references() is not a competing veto. The
			 *      survivor placement below reads PG_active and
			 *      PG_referenced, i.e. that function's own output, to
			 *      decide head promotion. Forcing removed the only path
			 *      that sets them from an access signal, so the
			 *      promotion this override was documented as backstopping
			 *      could not fire -- and the folios it then failed to
			 *      promote stayed in the oldest gen, making the override
			 *      look justified.
			 *
			 * A/B'd before removal rather than argued. QEMU, hot set
			 * sized to fit with a cold remainder going to swap, n=3 per
			 * arm with the order rotated: respecting references never
			 * failed to make progress (0 oom_kill, 3/3, against the
			 * 2026-05 failure this override was born to prevent), and
			 * showed no refault benefit either -- pswpin ranges overlap
			 * almost exactly (116013-304003 forced, 110925-307393 not).
			 * So this is a correctness change, not an optimisation: it
			 * stops suppressing VM_EXEC and the young-bit clear, and it
			 * deletes a certificate that could not be made honest. Do
			 * not re-introduce it as a performance measure.
			 */
			/*
			 * Throttle before isolating: if other concurrent
			 * reclaimers already have more isolated than the
			 * inactive list holds, wait for them to catch up
			 * instead of piling more folios into flight on top.
			 * One retry (mirrors shrink_inactive_list's
			 * `stalled` bool) -- but unlike an earlier version of
			 * this check, if it is STILL too many after that one
			 * wait, give up on this type for the rest of the
			 * sweep (drop to the per-type tail, same as the
			 * skip_file/ring-empty `break` above) rather than
			 * isolating another batch_max-sized batch (up to
			 * MARIE_PFN_SHRINK_BATCH pages) regardless.
			 * batch_max scales up to 256x SWAP_CLUSTER_MAX under
			 * heavy pressure, so "proceed anyway" lets every
			 * stalled reclaimer pile a full batch on top of an
			 * already-excessive isolated count instead of
			 * bailing out the way shrink_inactive_list's
			 * `stalled` path does (it returns 0 rather than
			 * isolating unconditionally). That mismatch is what
			 * let isolated_anon balloon into the tens-of-GB range
			 * (GitHub issue #6) even after the batch was already
			 * page-budget-capped: each concurrent reclaimer got
			 * one grace wait, then isolated up to its full batch
			 * regardless of whether the pileup had cleared.
			 * kswapd never blocks here (see
			 * marie_too_many_isolated).
			 */
			if (unlikely(marie_too_many_isolated(pgdat, type, sc))) {
				reclaim_throttle(pgdat, VMSCAN_THROTTLE_ISOLATED);
				if (fatal_signal_pending(current))
					goto done;
				if (unlikely(marie_too_many_isolated(pgdat, type, sc)))
					break;
			}

			/*
			 * Fill @folio_list, then call shrink_folio_list once.
			 *
			 * Scan writes candidate folios directly into
			 * scratch_batch[0..] in a SINGLE call: there is only
			 * one invocation per type per swept gen here (tier no
			 * longer exists as a scan dimension -- see state.h's
			 * byte-layout block), so the old CROSS-CALL "hard/soft
			 * ceiling" accumulator checks that used to bound
			 * repeated isolate calls within one sweep step do not
			 * have anything to accumulate against any more
			 * (n_taken / n_taken_pages are freshly reset to 0
			 * above every sweep_i iteration).
			 *
			 * The soft ceiling itself is NOT moot, though, and
			 * deleting it along with the accumulator was a
			 * regression: array_cap is a FOLIO count, so it alone
			 * lets one pass isolate 8192 folios -- 32 MiB at
			 * order-0 and far more with large folios -- at any
			 * priority, including the gentle first pass where
			 * marie_pfn_batch_threshold() asks for 256 pages. It
			 * is re-applied below in the claim loop, the one place
			 * that can still enforce it now that there is nothing
			 * between calls to enforce it in.
			 *
			 * Failed claims (try_get / test_clear_lru) leave
			 * the corresponding scratch_batch slot to be
			 * overwritten by the next successful claim --
			 * in-place compaction via accept_idx.
			 */
			{
				unsigned long nr_isolated, i;
				unsigned long accept_idx = n_taken;

				if (sc_reclaim_target_reached(sc))
					goto done;

				/*
				 * Bound the walk by the page budget as well as
				 * by the array. Every folio is at least one
				 * page, so batch_max folios can never yield
				 * fewer pages than batch_max; this only stops
				 * the bitmap walk from filling 8192 slots that
				 * the claim loop below will abandon after a
				 * few hundred pages.
				 */
				nr_isolated = marie_state_isolate_scan_l2lock(
					pgdat, type, sc_reclaim_idx(sc),
					&scratch_batch[n_taken],
					min(array_cap, batch_max),
					ULONG_MAX, oldest);

				for (i = 0; i < nr_isolated; i++) {
					/*
					 * Soft PAGE-count budget: cap the
					 * memory this pass takes off the LRU,
					 * not the folio count -- one THP is
					 * 512 pages, so a folio ceiling is not
					 * a bound on anything the caller asked
					 * for. batch_max is
					 * marie_pfn_batch_threshold(sc): 256
					 * pages at DEF_PRIORITY, scaling to
					 * array_cap only as priority falls.
					 *
					 * Checked before claiming rather than
					 * after, so a batch overshoots by at
					 * most the one folio that crosses the
					 * line. Folios left unclaimed in
					 * scratch_batch need no cleanup: the
					 * scan only stored pointers, and it is
					 * this loop that takes the reference,
					 * clears PG_lru and moves the
					 * counters.
					 */
					if (n_taken_pages >= batch_max)
						break;
					f = scratch_batch[n_taken + i];
					if (!folio_try_get(f))
						continue;
					if (!folio_test_clear_lru(f)) {
						folio_put(f);
						continue;
					}

					scratch_batch[accept_idx] = f;

					/*
					 * marie_evict_counters_only sets
					 * marie_state[pfn]'s ISOLATED bit
					 * (gating out every other GEN mutator
					 * -- walker, MADV_COLD, MADV_FREE,
					 * defrag restamp -- for the duration)
					 * and clears the (type, zone, gen)
					 * bitmap bit + decrements gen_occupied
					 * at that point. TRACKED stays set so
					 * install_local's early-out blocks any
					 * concurrent install from re-setting
					 * PG_lru while shrink_folio_list
					 * reclaims it. TRACKED (and ISOLATED)
					 * are wiped at the buddy handoff via
					 * marie_state_drop_pfn_at_free() for a
					 * reclaimed folio; survivors get
					 * marie_state[] refreshed and ISOLATED
					 * cleared in the putback loop below.
					 */
					marie_evict_counters_only(f);

					list_add(&f->lru, &folio_list);
					n_taken_pages += folio_nr_pages(f);
					accept_idx++;
				}
				n_taken = accept_idx;
			}

			if (!n_taken)
				break;

			/*
			 * PGSCAN accounting, mirroring upstream MGLRU's
			 * post-isolation bump (mm/vmscan.c evict_folios).
			 * n_taken is the count actually pulled off the LRU
			 * (the equivalent of MGLRU's `isolated`); upstream
			 * PGSCAN_* tracks isolated, not bitmap-scanned bits.
			 *
			 * NR_ISOLATED_ANON / _FILE must be bumped here so
			 * reclaim throttling and writeback congestion
			 * checks see Marie's in-flight isolation; the
			 * counter is decremented after shrink_folio_list
			 * finishes (whether the folio was reclaimed or put
			 * back).
			 */
			{
				mod_node_page_state(pgdat,
						    NR_ISOLATED_ANON + type,
						    n_taken_pages);
				marie_account_reclaim(lruvec, sc,
						PGSCAN_KSWAPD, PGSCAN_ANON,
						type, n_taken_pages);
			}

			/* ignore_references=false, always -- see the block above. */
			n_reclaimed = marie_shrink_folio_list(&folio_list, pgdat,
							sc, &stat, false,
							memcg);
			marie_harvest_second_chance(&stat, type);
			sc_add_reclaimed(sc, n_reclaimed);
			total_reclaimed += n_reclaimed;
			if (type < 2)
				atomic_long_add(n_reclaimed,
						&marie_dbg_reclaimed[type]);

			/*
			 * PGSTEAL accounting + matched NR_ISOLATED decrement.
			 * shrink_folio_list has either freed each folio or
			 * left it on @folio_list for putback; either way the
			 * isolation window for these n_taken folios is over.
			 */
			{
				mod_node_page_state(pgdat,
						    NR_ISOLATED_ANON + type,
						    -n_taken_pages);
				marie_account_reclaim(lruvec, sc,
						PGSTEAL_KSWAPD, PGSTEAL_ANON,
						type, n_reclaimed);
			}

			oldest_for_putback =
				marie_find_oldest_occupied_mlv(type);
			if (oldest_for_putback >= 0)
				putback_gen = (u8)((oldest_for_putback + 1)
					& (MARIE_PFN_NR_GENS - 1));
			else
				putback_gen = (u8)atomic_read(
					&marie_head_gen[type]);

			list_for_each_entry_safe(f, tmp, &folio_list, lru) {
				bool hot;
				u8 gen;
				struct lruvec *lv;
				unsigned long pfn;
				int zone;

				pfn = folio_pfn(f);
				/*
				 * Survivor placement signal: PG_active (hot) and
				 * PG_referenced (accessed since the last scan --
				 * the FOLIO_KEEP folios). NOT PG_workingset: that
				 * is the refault-on-ENTRY signal, consumed at
				 * install (marie_folio_install) to give a
				 * refaulted folio a protected re-entry; at putback
				 * the live access signal is referenced/active.
				 *
				 * A hot survivor is promoted to the head gen so it
				 * LEAVES the oldest (reclaim) gen rather than being
				 * re-isolated every round (tier's former "touched
				 * once" hysteresis is retired -- see state.h's
				 * byte-layout block -- so this is a direct binary
				 * decision now, not a carried-forward tier value).
				 * A referenced file folio stuck at the frontier is
				 * exactly what produces a perpetual non-IO FILE
				 * shortfall and the swappiness=1 anon nibble; head
				 * promotion drains the frontier so the FILE pass
				 * reaches the genuinely reclaimable (cold) file
				 * instead of conceding to anon. A cold survivor
				 * stays at oldest+1 (putback_gen).
				 *
				 * These two flags are folio_check_references()'s own
				 * output, which is why the force-reclaim override that
				 * used to skip that call could not coexist with this:
				 * it removed the only path that sets them from an
				 * access signal, leaving this promotion unable to fire
				 * for the folios it was meant to save. Nothing bounds
				 * a hot folio's escape from reclaim now except the
				 * reference check itself, which is self-limiting:
				 * folio_check_references() clears PG_referenced as it
				 * reads it, so a folio that is not being touched loses
				 * the flag on one pass and is reclaimed on the next.
				 */
				hot = folio_test_active(f) ||
				      folio_test_referenced(f);
				gen = hot
					? (u8)atomic_read(&marie_head_gen[type])
					: putback_gen;

				list_del_init(&f->lru);
				lv = folio_lruvec(f);
				zone = folio_zonenum(f);
				/*
				 * Normalize PG_active->0, mirroring
				 * marie_folio_install() and marie_evict_locked():
				 * shrink_folio_list's activate_locked path can leave
				 * PG_active set on a Marie-isolated folio, and a
				 * PG_active survivor going back into the ring would
				 * be treated as hot on its next pass.
				 *
				 * This no longer decides an accounting bucket. Both
				 * the credit below and every eventual debit take
				 * (lru, zone) from the state byte via marie_acct_lru
				 * / marie_acct_zone, so a stray PG_active can no
				 * longer land the survivor's +nr in ACTIVE_* while
				 * the debit comes out of INACTIVE_* -- the
				 * producer/consumer bucket split that used to
				 * underflow lru_zone_size. See account.h.
				 */
				if (folio_test_active(f))
					folio_clear_active(f);

				/*
				 * Survivor putback -- UNIFIED, global-only. The
				 * folio stays a Marie folio; we do NOT route it
				 * back through folio_putback_lru / folio_add_lru.
				 * That generic path re-enters the per-cpu
				 * folio_batch pipeline, which assumes legacy-LRU
				 * invariants (folio on a real list, counted in
				 * mz->lru_zone_size) that Marie folios break --
				 * under heavy pressure it freed still-dirty
				 * swapbacked folios out of the batch drain
				 * ("Bad page state").
				 *
				 * ISOLATED release (see the ISOLATED gate protocol
				 * in state.h): marie_evict_counters_only set
				 * ISOLATED and cleared this pfn's OLD (type, zone,
				 * gen) bitmap bit + gen_occupied at claim time, so
				 * there is nothing left to un-publish here -- only
				 * the NEW placement to publish. The byte is
				 * reconstructed from scratch (TRACKED | TYPE | ZONE
				 * | gen), which implicitly clears ISOLATED: with
				 * ISOLATED still held up to this exact write, no
				 * other mutator can be touching this pfn's byte or
				 * bitmap bit concurrently, so a plain store (not a
				 * CAS-retry) is safe. This closes the walker-vs-
				 * putback race the old bitmap-plane design was
				 * exposed to (a concurrent promote-on-access could
				 * roll this exact pfn to a new slot between isolate
				 * and putback, leaving a phantom gen_occupied count)
				 * by preventing every other mutator from touching
				 * this pfn's GEN at all during the isolate/putback
				 * window, rather than reconciling after the fact.
				 */
				if (pfn < marie_state_size) {
					u8 new_b = MARIE_PFN_TRACKED |
						(type ? MARIE_PFN_TYPE_FILE : 0) |
						marie_pfn_zone_bits(zone) |
						((u8)gen << MARIE_PFN_GEN_SHIFT);
					u8 old_b = xchg(&marie_state[pfn], new_b);

					marie_gen_occ_settle(pfn, old_b, new_b);
					/* Scan index; return value advisory. */
					marie_bm_set(&marie_track_bm[type][zone][gen], pfn);

					/*
					 * Account the survivor's
					 * re-installation. This publish clears
					 * ISOLATED, so pred goes 0 -> 1 and the
					 * +1 / +nr credit is DERIVED from that
					 * transition. Clearing ISOLATED is also
					 * what hands the "is a credit
					 * outstanding" answer back to every
					 * downstream teardown path, which is why
					 * the freed branch below must undo the
					 * whole publish and not just the
					 * counters. See account.h.
					 */
					marie_acct_settle_isolate(lv,
						folio_nr_pages(f), old_b, new_b);
				}

				if (!folio_put_testzero(f)) {
					/*
					 * Isolation ref dropped, folio still alive.
					 * Set PG_lru so the next scan can re-isolate
					 * it via folio_test_clear_lru.
					 */
					folio_set_lru(f);
				} else {
					/*
					 * Isolation ref was the last one -- folio is
					 * being freed now. PG_lru is clear (was cleared
					 * at isolation), so __folio_put's
					 * __page_cache_release will not call
					 * del_page_from_lru_list and will not debit
					 * mz->lru_zone_size a second time -- isolation
					 * already debited it (the install +nr is settled
					 * by the isolate path), so a free-time debit here
					 * would underflow.
					 *
					 * shrink_folio_list's activate_locked path may
					 * set PG_active on a folio whose PG_lru is
					 * already clear (Marie isolated it). Normally
					 * PAGE_FLAGS_CHECK_AT_FREE is satisfied because
					 * folio_activate() checks PG_lru and is a no-op
					 * when it is clear -- but some stock paths set
					 * PG_active directly (e.g. folio_set_active in
					 * the deactivate batch). Clear it here; the
					 * folio has no live references and is not on any
					 * LRU list, so clearing PG_active is safe.
					 *
					 * Undo the ENTIRE putback publish before
					 * completing the free -- state byte, scan
					 * bit, gen_occupied and counters -- via the
					 * one primitive that does all four as a
					 * single derived transition.
					 *
					 * Undoing only the COUNTERS here was a real
					 * premature-OOM bug. The publish a few lines
					 * above rebuilt marie_state[pfn] as
					 * (TRACKED | TYPE | ZONE | gen), which CLEARS
					 * ISOLATED -- and TRACKED && !ISOLATED is
					 * exactly the predicate every teardown path
					 * reads as "a credit is still outstanding".
					 * __folio_put() below runs
					 * mem_cgroup_uncharge() -> uncharge_folio() ->
					 * lru_marie_uncharge_backstop() BEFORE the
					 * buddy handoff, so a byte left saying
					 * TRACKED && !ISOLATED made the backstop
					 * settle a FULL second debit on a folio this
					 * branch had already settled.
					 * (__folio_put's other Marie exit,
					 * __page_cache_release ->
					 * lru_marie_release_folio, is gated on
					 * folio_test_lru() and bails -- this branch
					 * never sets PG_lru -- so the backstop was the
					 * sole second debiter.)
					 *
					 * That double-debit was cumulative and
					 * unbounded: marie_nr_folios went negative and
					 * NR_INACTIVE_ANON/_FILE walked down by @nr per
					 * occurrence until they floored at 0, so
					 * resident anon stopped being accounted
					 * anywhere. vmscan, marie_too_many_isolated,
					 * marie_file_floor_protect and
					 * marie_node_under_pressure all size reclaim
					 * from those counters, so reclaim concluded
					 * there was nothing left to reclaim and OOM'd
					 * with GBs still resident. Observed on real
					 * hardware as nr_folios -294350 with
					 * Inactive(anon) at exactly 0 kB while anon
					 * reclaim was churning, ~24 GiB unaccounted for
					 * by any counter -- the same signature
					 * mm/swap.c's __page_cache_release comment
					 * records for the TRACKED-only-gate bug on the
					 * isolate path. That site was fixed; this one
					 * was missed.
					 *
					 * It is now closed by construction rather than
					 * by remembering to undo two things: the debit
					 * is DERIVED from the byte transition
					 * (account.h), so the backstop's later attempt
					 * on an already-cleared byte computes a delta
					 * of zero. Forgetting the byte wipe could no
					 * longer double-count; it would only leave a
					 * stale scan bit.
					 */
					folio_clear_active(f);
					marie_state_drop_pfn_isolate(f);
					__folio_put(f);
				}
			}

			/*
			 * No deferred drop pass: the scan-bitmap slot was
			 * retired at isolate (counters_only), and the TRACKED
			 * byte of a reclaimed folio is wiped at its buddy
			 * handoff (marie_state_drop_pfn_at_free via the
			 * free_pages_prepare hook). Folios still alive in
			 * folio_list went through the survivor putback above,
			 * which re-published a fresh scan slot via
			 * marie_state_move_to_gen.
			 */

			/*
			 * Gen visited; loop to the next-oldest. The sweep ends
			 * by `break` (ring empty / nothing isolatable) or by the
			 * MARIE_PFN_NR_GENS bound, then falls to the per-type
			 * tail. `goto done` already left on target-reached.
			 */
			}

			/*
			 * Per-iteration tail.
			 *
			 * Bias controller update is skipped when:
			 *   - !attempted_pick: external override (skip_file
			 *     from clean_min_ratio) blocked the scan. The
			 *     bias must track actual picking policy, not
			 *     policy preempted before it ran.
			 *   - skip_file is in effect for THIS call: even
			 *     the ANON pick that succeeds during a
			 *     skip_file regime is happening only because
			 *     file was forcibly removed from contention.
			 *     Freezing the controller during the override
			 *     keeps the bias at its pre-override value, so
			 *     when file recovers above clean_min_ratio the
			 *     proportional regime resumes without an
			 *     overshoot driven by anon-only reclaim that
			 *     was never about the swappiness ratio.
			 *
			 * swappiness=1 (FILE_THEN_ANON) depletion-fallback
			 * gate (see the tail `if` below). Two independent
			 * reasons divert reclaim to ANON, on separate layers:
			 *
			 *   1. file < clean_min_ratio floor: handled UPFRONT by
			 *      marie_file_floor_protect -> skip_file -> pick
			 *      ANON_STRICT. Protects a minimum clean-file
			 *      reserve and never reaches here (skip_file
			 *      short-circuits FILE_THEN_ANON).
			 *
			 *   2. file >= floor but file reclaim cannot keep pace:
			 *      detected HERE by the FILE pass FAILING TO MEET
			 *      this call's reclaim target. A target-meeting FILE
			 *      pass exits via the tier loop's
			 *      sc_reclaim_target_reached() -> `goto done`, PAST
			 *      this tail; so merely arriving here means file fell
			 *      short. Occupancy/tier cannot tell reclaimability
			 *      or throughput apart -- a tracked file folio may be
			 *      hot/dirty/mapped, and how much actually frees is
			 *      known only by trying (shrink_folio_list). The
			 *      earlier gate keyed on the FILE pass returning
			 *      EXACTLY zero, which conflates "no reclaimable
			 *      file" with "file frees a positive trickle that
			 *      cannot match the allocation rate": while any
			 *      recyclable clean pagecache keeps cycling (refault /
			 *      IO refill) the FILE pass returns >0 forever, anon
			 *      is never scanned, and GBs of swappable anon OOM
			 *      with swap free. Sufficiency, not exact-zero, is the
			 *      correct depletion signal.
			 *
			 *      Because the FILE pass now SWEEPS the whole aged
			 *      gen ring before arriving here, merely arriving
			 *      means file fell short after reclaiming everything
			 *      reclaimable this call. The remedy depends only on
			 *      the floor: while clean file is still ABOVE
			 *      clean_min_ratio, DEFER -- successive file-only
			 *      calls drain it: cold clean file scores
			 *      FOLIOREF_RECLAIM_CLEAN, so it drains with no
			 *      reference override, and a pass that still falls
			 *      short escapes through the free/refault/memcg
			 *      terms below. Once file is at
			 *      the floor, CONCEDE to anon -- the file reserve is
			 *      protected and swapping anon is the OOM-with-swap-
			 *      free safety; the pick flips to ANON_STRICT next
			 *      call.
			 *
			 * `goto done` (target reached inside the tier loop)
			 * jumps PAST this tail intentionally: we are winning,
			 * the controller does not need a back-pressure tick.
			 */
			/*
			 * anon_unreclaimable forced FILE_STRICT above,
			 * bypassing the proportional controller; do not let
			 * those forced-file picks drive the bias (matches the
			 * "special swappiness values bypass the controller"
			 * rule -- the bias must resume cleanly once swap
			 * capacity returns and can_reclaim_anon flips back).
			 *
			 * The FILE_THEN_ANON depletion fallback (idx==1 ANON,
			 * reached only because the FILE pass found nothing
			 * reclaimable) is likewise a forced pick driven by file
			 * depletion, not by the swappiness ratio, so it must not
			 * drive the bias either.
			 */
			if (attempted_pick && !skip_file && !anon_unreclaimable &&
			    !(pick_kind == MARIE_PICK_FILE_THEN_ANON && idx == 1) &&
			    likely(!oom_victim))
				marie_swap_bias_update(type,
						       total_reclaimed, swappiness);
			if (likely(!oom_victim) &&
			    pick_kind == MARIE_PICK_FILE_THEN_ANON &&
			    idx == 0 && !skip_file) {
				/*
				 * swappiness=1 depletion fallback -- the FILE pass
				 * SWEPT the whole aged gen ring this call and still
				 * fell short (a target-meeting pass left via the tier
				 * loop's sc_reclaim_target_reached() -> `goto done`,
				 * PAST this gate). Decide anon purely on the floor:
				 *
				 *   file still ABOVE the clean_min_ratio floor ->
				 *   DEFER. swappiness=1 drains file to the floor before
				 *   anon; the per-call sweep is batch-capped, so a
				 *   large target is met across successive file-only
				 *   calls, not by conceding to anon while reclaimable
				 *   file sits above the floor. This is not a rotation
				 *   treadmill: cold clean file scores
				 *   FOLIOREF_RECLAIM_CLEAN and drains without a
				 *   reference override, referenced file that the walker
				 *   has already judged is force-reclaimed by the epoch
				 *   backstop above, and anything still short escapes
				 *   via the free and refault terms rather than by
				 *   force-evicting mapped file. memcg-targeted reclaim
				 *   that made no file progress is the exception -- it
				 *   concedes so the limited memcg makes progress.
				 *
				 *   file at/below the floor -> CONCEDE to anon. The
				 *   file reserve is protected at the floor and swapping
				 *   anon is the OOM-with-swap-free safety. Reaching
				 *   FILE_THEN_ANON proved anon is reclaimable, so swap
				 *   capacity exists by construction.
				 *
				 *   node under acute free pressure (free <= 2*high) ->
				 *   CONCEDE regardless of the file level. The DEFER above
				 *   assumes file-only calls drain clean file to the floor;
				 *   a workload refilling clean file above the floor faster
				 *   than the batch-capped sweep drains it holds the floor
				 *   forever unreached, so this tail fires every time the
				 *   aged FILE ring depletes while free stays pinned at the
				 *   watermarks and GBs of swappable anon (incl. swap-backed
				 *   shmem) never reach swap -> premature OOM with swap free.
				 *   Once free is at the watermarks the file level is moot;
				 *   offer anon now. See marie_node_under_pressure().
				 *
				 * Equivalent original terms retained: floor_protect, and the
				 * memcg-targeted no-file-progress concede; the pressure term
				 * is the added early treadmill escape.
				 */
				bool floor  = marie_file_floor_protect(pgdat);
				bool freep  = marie_node_under_pressure(pgdat);
				bool refp   = marie_file_refaulting();
				bool memcgp = sc_cgroup_reclaim(sc) &&
					      !total_reclaimed;
				bool concede = floor || freep || refp || memcgp;

				if (!concede)
					goto done;

				/* Trigger attribution (priority floor>free>refault>memcg). */
				atomic_long_inc(&marie_dbg_concede[
					floor ? 0 : freep ? 1 : refp ? 2 : 3]);

				/*
				 * Engage the ANON pass NOW. If a final FILE batch
				 * happened to tip the target, the idx==1 ANON pass
				 * self-aborts at its own sc_reclaim_target_reached()
				 * gate, so no anon is over-reclaimed.
				 */
				drain_mask |= MARIE_DRAIN_ANON;
			}
		}
done:
		if (using_percpu)
			atomic_set(&buf->in_use, 0);
	}

	{
		unsigned long got = sc_nr_reclaimed(sc) - budget_done;
		long over = (long)got - (long)budget_need;

		atomic_long_add(budget_need, &marie_dbg_budget[0]);
		atomic_long_add(got, &marie_dbg_budget[1]);
		atomic_long_inc(&marie_dbg_budget[2]);
		if (over > 0) {
			long prev = atomic_long_read(&marie_dbg_budget[3]);

			while (over > prev &&
			       !atomic_long_try_cmpxchg(&marie_dbg_budget[3],
							&prev, over))
				;
		}
	}

	return drain_mask;
}
