/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_PFN_INSTALL_H
#define _MM_LRU_MARIE_PFN_INSTALL_H

#include <linux/atomic.h>
#include <linux/memcontrol.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>

#include "bitmap.h"
#include "state.h"

/*
 * Marie's "publish a PFN as TRACKED" primitive, factored out of the
 * install/split paths.
 *
 * What it writes (the single sources of truth for "Marie owns this PFN"
 * and "where is it in aging space"):
 *   - marie_state[pfn]: TRACKED | (type) | (zone) | (gen) -- one byte,
 *     identity and freshness merged (state.h's byte-layout block)
 *   - marie_bm_set(&marie_track_bm[type][zone][gen], pfn): the exact
 *     scan-acceleration bit for this PFN's (type, zone, gen)
 *   - marie_gen_occ_inc(pfn, gen, type): the single gen-occupancy choke-point
 *     (state.h), shared with every other transition site; @pfn also drives
 *     the Marie defrag per-pageblock occupancy mirror
 *
 * What it deliberately does NOT touch:
 *   - folio->flags (PG_active / PG_lru) -- the install path flips these
 *     in one atomic mask write after publish; the split path's caller
 *     sets PG_lru later.
 *   - folio->lru list pointers -- INIT_LIST_HEAD vs list_add_tail differs
 *     between install and split.
 *   - marie_nr_folios and vmstat lru_size -- accounted by the caller (or
 *     by marie_folio_install for the fresh-install path).
 *   - marie_gen_installs -- the global install "throttle" counter that
 *     drives gen advance; split intentionally does NOT bump it because
 *     the split tail inherits its parent's install budget (the parent
 *     was already counted at fault-install).
 *
 * Caller context: lru_lock held with IRQs off.
 *
 * RETURNS the published byte and reports the replaced one via @old_out, so
 * the caller can hand the (old -> new) transition to marie_acct_settle_*()
 * and let the counter delta be DERIVED from it (see account.h).
 *
 * A PLAIN read + store, deliberately not an xchg. This is the hottest path
 * in the subsystem -- one call per folio entering the LRU -- and an
 * exactly-once RMW buys nothing here, because publish is already
 * exactly-once for a different reason: installs on one PFN are serialised by
 * lru_lock, and marie_folio_install's "already TRACKED" early-out (under
 * that same lock) rejects the second one. The byte cannot be TRACKED on
 * entry, so pred(old) is 0 by construction and no other mutator can be
 * racing: isolate needs a successful folio_test_clear_lru and PG_lru is not
 * set until after this returns; marie_state_move_to_gen needs TRACKED, which
 * only this store sets; and the free-side teardown paths need an allocated
 * page to be on its way out. Teardown is where the concurrent claimants
 * genuinely are (evict vs uncharge_backstop vs the buddy-handoff hook), and
 * that is where the atomic RMW lives (marie_state_untrack).
 *
 * Dropping the xchg also drops the barrier it incidentally provided, so note
 * where the publish ordering actually comes from -- both remaining observers
 * are gated behind an atomic that follows this store:
 *
 *   - the scanner only reads the byte for PFNs whose index bit is set, and
 *     marie_bm_set() below is a test_and_set_bit AFTER the store, so
 *     observing the bit implies observing the byte;
 *   - the del side is gated on PG_lru, which marie_folio_install sets via
 *     set_mask_bits() -- a try_cmpxchg loop, fully ordered on success -- so
 *     PG_lru can never become visible before TRACKED.
 *
 * Both hold on weakly-ordered architectures too, not just x86-TSO. A
 * lock-free marie_state_move_to_gen that happens to read the pre-publish
 * value simply sees !TRACKED and declines, costing one missed promotion.
 *
 * WRITE_ONCE/READ_ONCE rather than bare assignment: other CPUs read this
 * byte lock-free (marie_state_move_to_gen and the scanner both use
 * READ_ONCE). A single byte cannot tear, but the annotation keeps the
 * compiler from moving, merging or duplicating the access, and keeps KCSAN
 * quiet. It adds no barrier and no instructions.
 */
static inline u8 marie_pfn_publish_inherit(struct folio *f, int type,
					   u8 gen, int zone, u8 *old_out)
{
	unsigned long pfn = folio_pfn(f);
	u8 new_b = MARIE_PFN_TRACKED |
		(type ? MARIE_PFN_TYPE_FILE : 0) |
		marie_pfn_zone_bits(zone) |
		((u8)gen << MARIE_PFN_GEN_SHIFT);

	*old_out = READ_ONCE(marie_state[pfn]);
	WRITE_ONCE(marie_state[pfn], new_b);
	marie_gen_occ_settle(pfn, *old_out, new_b);
	/* Scan index, derived from the byte above; return value is advisory. */
	marie_bm_set(&marie_track_bm[type][zone][gen], pfn);
	return new_b;
}

/*
 * marie_pfn_publish_isolated - publish a PFN as TRACKED *and ISOLATED*:
 * marie_state[] is written, but marie_track_bm is not touched and
 * gen_occupied is not incremented.
 *
 * This is the correct publish for a reclaim-split THP tail, and mirrors
 * exactly the state marie_evict_counters_only leaves on an isolated folio
 * -- via the explicit ISOLATED bit (state.h), not an implicit "TRACKED
 * but no scan bit anywhere" encoding. That implicit encoding was the root
 * of a real historical freeze (see below); making ISOLATED explicit here
 * removes the ambiguity it lived in.
 *
 * The parent THP was isolated via marie_evict_counters_only (gen_occupied
 * debited, bitmap bit cleared, ISOLATED set, nr_folios/lru_size debited)
 * BEFORE shrink_folio_list split it, so each child must enter the
 * isolated state, NOT a fresh install:
 *   - a child that is reclaimed frees with ISOLATED already set -> nothing
 *     to retire beyond the byte wipe, net-zero, matching the parent's
 *     already-debited aggregate;
 *   - a child that survives is balanced EXACTLY ONCE by the putback
 *     release path (republishes marie_state[], gen_occupied +1, bitmap
 *     bit set, ISOLATED cleared; the derived credit adds
 *     nr_folios + lru_size) -- the same balance as a non-split isolated
 *     survivor.
 *
 * Using marie_pfn_publish_inherit here instead publishes the tail in the
 * fresh-install state (gen_occupied + bitmap + nr_folios), so a surviving
 * tail is counted twice (split inc + putback inc) but retired once at
 * free, leaking +1 gen_occupied (and +1 nr_folios) per surviving tail.
 * After the tails free, that residue is phantom gen_occupied>0 occupancy
 * with nothing real behind it: find_oldest keeps returning the phantom
 * gen, the scanner finds nothing there, anon reclaim makes zero progress,
 * and head-advance is wedged (its gate is gen_occupied[next]==0).
 * THP-only (order-0 never splits) and persists across OOM until a reboot
 * zeroes the global gen ring -- the "first tail run reclaims, every retry
 * stalls at swapout onset" freeze.
 *
 * Caller context: identical to marie_pfn_publish_inherit (split path holds
 * the per-type lock; the tail is exclusively owned, off-LRU, PG_lru clear).
 */
static inline u8 marie_pfn_publish_isolated(struct folio *f, int type,
					    u8 gen, int zone, u8 *old_out)
{
	unsigned long pfn = folio_pfn(f);
	u8 new_b = MARIE_PFN_TRACKED | MARIE_PFN_ISOLATED |
		(type ? MARIE_PFN_TYPE_FILE : 0) |
		marie_pfn_zone_bits(zone) |
		((u8)gen << MARIE_PFN_GEN_SHIFT);

	/*
	 * Returns the replaced byte for symmetry with
	 * marie_pfn_publish_inherit. A reclaim-split tail's own PFN byte was
	 * never individually published (only the compound head's was), so the
	 * transition is 0 -> (TRACKED|ISOLATED): pred stays 0 on both sides
	 * and marie_acct_settle_*() derives d == 0, which is exactly the
	 * "no scan bit, no gen_occupied, no nr_folios" contract above. Routing
	 * it through settle anyway means that if this ever ran on a PFN that
	 * DID hold a credit, the credit would be retired rather than lost.
	 *
	 * Plain store, for the same reason as marie_pfn_publish_inherit: the
	 * split path holds the per-type lock and the tail is exclusively owned
	 * (off-LRU, PG_lru clear, its own PFN byte never individually
	 * published), so there is no concurrent mutator for an RMW to exclude.
	 */
	*old_out = READ_ONCE(marie_state[pfn]);
	WRITE_ONCE(marie_state[pfn], new_b);
	return new_b;
}

/*
 * marie_folio_install - the unified fresh-install path.
 *
 * Single entry point that replaces the former marie_install_local /
 * marie_install_locked pair. Both call sites (lru_marie_add_folio for THP
 * via per-type lock + small folio direct, and marie_change_state_lruvec
 * during gate-on fill) now route here. The per-type lock context that
 * used to distinguish "locked" from "local" is the caller's concern, not
 * this function's: the body only requires lru_lock + IRQs off and uses
 * the same publish + flag flip + account sequence in both cases.
 *
 * Sequence:
 *   1. TRACKED early-out (returns false). Defends against gate-flip race
 *      and reclaim-survivor re-install (TRACKED is preserved across
 *      isolate by design; the putback release path in state.c handles
 *      the survivor putback separately, never this function).
 *   2. Capture PG_active and clear it early -- a fresh install always
 *      lands on the head gen regardless of any hotness signal (see
 *      state.h's byte-layout block), so PG_active is normalised purely
 *      for inst_lru/accounting correctness, not to seed anything.
 *   3. INIT_LIST_HEAD(&f->lru) -- a recycled folio arrives with
 *      LIST_POISON{1,2} that would later fault list_del_init.
 *   4. Publish per-PFN state via marie_pfn_publish_inherit.
 *   5. Bump marie_gen_installs[type] and advance the head at
 *      marie_gen_growth_live[type]. Split path skips this bump
 *      (publish_inherit only).
 *   6. set_mask_bits(PG_active->0, PG_lru->1) -- one atomic flag write.
 *      Ordered AFTER step 4 so a concurrent __page_cache_release
 *      observing PG_lru=1 also observes TRACKED=1.
 *   7. Account (marie_nr_folios + vmstat lru_size, i.e. NR_LRU_BASE /
 *      NR_ZONE_LRU_BASE) -- DERIVED from the publish transition via
 *      marie_acct_settle_locked, never stated. See account.h.
 *
 * Returns true on success, false on TRACKED early-out.
 */
bool marie_folio_install(struct folio *f);

#endif /* _MM_LRU_MARIE_PFN_INSTALL_H */
