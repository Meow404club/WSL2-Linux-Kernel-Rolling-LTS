/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_ACCOUNT_H
#define _MM_LRU_MARIE_ACCOUNT_H

#include <linux/lockdep.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>

#include "state.h"

/*
 * Marie's global counters, and the ONE rule that keeps them exact.
 *
 * Marie holds zero per-lruvec state, so the only counters are global: the
 * percpu marie_nr_folios (+-1 per folio) and the vmstat lru_size pair via
 * marie_update_lru_size / __update_lru_size (both maintain NR_LRU_BASE and
 * the node-global NR_ZONE_LRU_BASE by the same amount, so a credit taken
 * through one and a debit through the other still cancel).
 *
 *
 * THE ACCOUNTING INVARIANT
 * ------------------------
 *
 *   marie_nr_folios == #{ pfn : marie_acct_pred(marie_state[pfn]) }
 *
 * where marie_acct_pred() is "this PFN holds an outstanding +1", i.e.
 * TRACKED && !ISOLATED. lru_size holds the matching page-weighted sum.
 * The state byte is the single source of truth; the counters are a
 * projection of it.
 *
 * Marie's byte encoding already expresses this exactly, with no spare bit
 * needed -- the two lifecycle bits ARE the accounting ledger:
 *
 *   TRACKED=1, ISOLATED=0   +1 outstanding      (installed, or put back)
 *   TRACKED=1, ISOLATED=1   settled             (reclaim isolate claimed it)
 *   TRACKED=0               settled             (evicted / freed / never in)
 *
 *
 * WHY THE DELTA IS DERIVED, NEVER STATED
 * --------------------------------------
 *
 * The counters are updated ONLY by marie_acct_settle_*(), which takes the
 * @old and @new bytes of an ALREADY-COMMITTED atomic state transition and
 * applies d = pred(new) - pred(old). A caller cannot ask for a credit or a
 * debit; it can only report a transition, and the delta follows from it.
 *
 * That makes double-counting unrepresentable. Two paths racing to tear the
 * same folio down both call settle, but only the one whose atomic RMW
 * actually committed the 1->0 predicate transition sees d = -1; the other
 * observes pred(old) == 0 and computes d = 0. Any number of teardown
 * attempts therefore produce exactly one debit -- the same
 * transition-gating discipline that marie_bm_set/marie_bm_clear already
 * give gen_occupied, and the reason gen_occupied has never drifted while
 * hand-written +-1 pairs did.
 *
 * This is not a micro-optimisation of the old helpers; it is what closes a
 * confirmed premature-OOM bug. The survivor-putback path publishes a fresh
 * byte that CLEARS ISOLATED, which under the old scheme silently
 * re-enabled lru_marie_uncharge_backstop()'s independent debit on top of
 * the putback's own, walking marie_nr_folios to -294350 and pinning
 * Inactive(anon) at 0 kB while GBs stayed resident: reclaim then sized
 * itself from destroyed counters and OOM'd early. With the delta derived
 * from the transition, that second debit computes to zero by construction.
 *
 *
 * BUCKET AND ZONE COME FROM THE BYTE TOO
 * --------------------------------------
 *
 * marie_acct_lru()/marie_acct_zone() reconstruct the (lru, zone) bucket
 * from the state byte rather than from folio flags or folio_zonenum():
 *
 *   - lru: install normalises PG_active to 0 before computing its index,
 *     so every credit lands in INACTIVE_*; TYPE alone determines the
 *     bucket. Reading folio_lru_list() at debit time is unsafe because
 *     shrink_folio_list may have re-stamped PG_active on an isolated
 *     folio, which would debit ACTIVE_* -- a bucket Marie never credited.
 *
 *   - zone: the byte's ZONE field is a 2-bit MASK of folio_zonenum(), not
 *     a clamp, so it is lossy for zones >= MARIE_PFN_NR_ZONES_ENCODED
 *     (ZONE_MOVABLE, ZONE_DEVICE). Deriving BOTH the credit and the debit
 *     from the byte makes them cancel regardless: a folio credited under
 *     the encoded zone is debited under the same encoded zone. Crediting
 *     with the raw zone and debiting with the decoded one would inflate
 *     ZONE_MOVABLE's NR_ZONE_LRU_BASE forever and drive ZONE_DMA's
 *     negative on any kernel with a populated movable zone.
 *
 *
 * TWO CONTEXTS
 * ------------
 *
 *   LOCKED   - caller holds lv->lru_lock with IRQs off (install,
 *              evict_locked). Asserts both via lockdep.
 *
 *   ISOLATE  - caller holds NOTHING, IRQs on (reclaim isolate, survivor
 *              putback). Owns local_irq_save/restore so marie_pc_add's
 *              fast path and __mod_zone_page_state are safe against
 *              same-CPU softirq reentrancy (the property 9c6a93782
 *              introduced).
 */

/*
 * marie_acct_pred() -- "this byte holds an outstanding +1" -- lives in
 * state.h next to the byte layout, because gen_occupied derives from the
 * same predicate (marie_gen_occ_settle). The counters here are its total;
 * gen_occupied is its per-(gen, type) partition.
 */

/* The bucket a Marie credit always lands in; see the header block. */
static inline enum lru_list marie_acct_lru(u8 b)
{
	return (b & MARIE_PFN_TYPE_MASK) ? LRU_INACTIVE_FILE
					 : LRU_INACTIVE_ANON;
}

/* The ENCODED zone the credit was taken under; see the header block. */
static inline enum zone_type marie_acct_zone(u8 b)
{
	return (enum zone_type)((b & MARIE_PFN_ZONE_MASK) >>
				MARIE_PFN_ZONE_SHIFT);
}

/*
 * Shared body. @nr is the page weight for lru_size; pass 0 to move
 * marie_nr_folios alone (the THP-split tail case, where the parent's
 * install already credited every page of the pre-split compound and only
 * the folio COUNT changes).
 *
 * The bucket is taken from whichever side of the transition held the
 * credit, so a credit and its eventual debit always name the same
 * (lru, zone) slot even if other fields of the byte changed in between.
 */
static inline int marie_acct_delta(u8 old, u8 new_b, u8 *held)
{
	int d = marie_acct_pred(new_b) - marie_acct_pred(old);

	*held = (d > 0) ? new_b : old;
	return d;
}

/*
 * LOCKED context: settle the transition @old -> @new_b for a folio of
 * @nr pages. Caller holds lv->lru_lock, IRQs off.
 */
static inline void marie_acct_settle_locked(struct lruvec *lv, long nr,
					    u8 old, u8 new_b)
{
	u8 held;
	int d = marie_acct_delta(old, new_b, &held);

	if (!d)
		return;

	lockdep_assert_held(&lv->lru_lock);
	lockdep_assert_irqs_disabled();

	marie_pc_add(&marie_nr_folios, d);
	marie_update_lru_size(lv, marie_acct_lru(held), marie_acct_zone(held),
			      d * nr);
}

/*
 * ISOLATE context: settle the transition @old -> @new_b for a folio of
 * @nr pages. Caller holds no lock, IRQs on.
 */
static inline void marie_acct_settle_isolate(struct lruvec *lv, long nr,
					     u8 old, u8 new_b)
{
	unsigned long flags;
	u8 held;
	int d = marie_acct_delta(old, new_b, &held);

	if (!d)
		return;

	WARN_ON_ONCE(irqs_disabled());

	local_irq_save(flags);
	marie_pc_add(&marie_nr_folios, d);
	__update_lru_size(lv, marie_acct_lru(held), marie_acct_zone(held),
			  d * nr);
	local_irq_restore(flags);
}

/*
 * marie_acct_settle_count - marie_nr_folios only, no lruvec required.
 *
 * For the one teardown point that cannot resolve a lruvec:
 * marie_state_drop_pfn_at_free() runs from free_pages_prepare, after
 * memcg_data has been zeroed, so folio_memcg is unsafe to dereference
 * there. A folio reaching buddy still holding its credit (never charged,
 * hence never uncharged, hence never seen by
 * lru_marie_uncharge_backstop) is settled here for the folio count --
 * which is the counter that reclaim's own gates and the OOM path read --
 * and its lru_size page weight is the one residue Marie cannot resolve.
 * IRQ-tolerant: the free path runs from any context.
 */
static inline void marie_acct_settle_count(u8 old, u8 new_b)
{
	unsigned long flags;
	u8 held;
	int d = marie_acct_delta(old, new_b, &held);

	if (!d)
		return;

	local_irq_save(flags);
	marie_pc_add(&marie_nr_folios, d);
	local_irq_restore(flags);
}

#endif /* _MM_LRU_MARIE_ACCOUNT_H */
