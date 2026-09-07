// SPDX-License-Identifier: GPL-2.0
/*
 * Marie folio-facing hook surface: install/evict, the mark_accessed /
 * deactivate / rotate / activate / lazyfree family, the per-type
 * drain-exclusion primitives, and the module's final counter/lock
 * init. This is the API the rest of mm/ (vmscan.c, swap.c,
 * memcontrol.c via core.c) actually calls into — see pfn_install.h
 * for the install/evict contract this file implements.
 */

#define pr_fmt(fmt) "marie_state: " fmt

#include <linux/atomic.h>
#include <linux/bitops.h>
#include <linux/jump_label.h>
#include <linux/lru_marie.h>
#include <linux/memcontrol.h>
#include <linux/minmax.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>
#include <linux/mmzone.h>
#include <linux/percpu.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/swap.h>
#include <linux/vmstat.h>

#include "../internal.h"
#include "state_compat.h"
#include "account.h"
#include "pfn_install.h"
#include "state.h"

/* --- install / evict implementations --- */


static DEFINE_PER_CPU(int[ANON_AND_FILE], marie_drain_depth);

void marie_drain_enter_type(int type)
{
	this_cpu_inc(marie_drain_depth[type]);
}
void marie_drain_exit_type(int type)
{
	this_cpu_dec(marie_drain_depth[type]);
}
bool marie_in_drain_type(int type)
{
	return this_cpu_read(marie_drain_depth[type]) > 0;
}

/*
 * ---------------------------------------------------------------------
 *  Install / evict — direct synchronous transitions under lru_lock
 * ---------------------------------------------------------------------
 *
 * The per-PFN paradigm reduces every Marie folio's state to a single
 * bit (TRACKED in marie_state[pfn]). There are exactly two state
 * transitions:
 *
 *   marie_folio_install:   TRACKED 0 -> 1   (writes gen, type, zone,
 *                          sets PG_lru, bumps counters; defined below,
 *                          declared in pfn_install.h)
 *   marie_evict_locked:    TRACKED 1 -> 0   (counter decrements +
 *                          per-PFN state wipe via marie_state_drop_pfn)
 *
 * Both are called with the caller's lru_lock irqsave held, so the
 * per-PFN byte write, the bitmap mutations, and the counter updates all
 * run in the same atomic context. PG_active hygiene and other
 * cross-cutting concerns are concentrated here.
 */


/*
 * marie_folio_install - unified fresh install (TRACKED 0 -> 1).
 *
 * Replaces the former marie_install_local / marie_install_locked pair.
 * The two used to differ only in the order of (publish, account, flag)
 * and in the PG_lru set method; this canonical form picks set_mask_bits
 * (atomic PG_active clear + PG_lru set in one mask write) and the
 * publish -> flag -> account order from install_local.
 *
 * Call site:
 *   - lru_marie_add_folio (THP under per-type lock, small folio direct)
 *
 * Per-type lock is a property of the caller, not of this function: the
 * body only requires lru_lock + IRQs off and behaves identically whether
 * or not the caller additionally holds the per-type lock.
 *
 * Returns true on success, false on the "already TRACKED" early-out.
 * See pfn_install.h for the contract documentation.
 */
bool marie_folio_install(struct folio *folio)
{
	struct lruvec *lv = folio_lruvec(folio);
	bool was_active;
	int type, zone;
	u8 head, old_b, new_b;
	unsigned long pfn;

	lockdep_assert_held(&lv->lru_lock);
	lockdep_assert_irqs_disabled();

	/*
	 * "Already TRACKED" early-out. A folio reaching install while its
	 * per-PFN byte is still TRACKED is a Marie-owned, reclaim-isolated
	 * folio (the deferred-teardown design preserves TRACKED while PG_lru
	 * is cleared) being re-added through a path that lacks a TRACKED gate
	 * -- e.g. folio_add_lru()/folio_putback_lru() on an anon folio that
	 * reclaim isolated into the swap cache and a fault then swaps back in.
	 * Re-installing would re-set PG_lru and double-count Marie's counters;
	 * the resurrected PG_lru then survives onto the buddy free path and
	 * trips "Bad page state |lru|" PAGE_FLAGS_CHECK_AT_FREE. Bail so the
	 * in-flight reclaim retains ownership.
	 *
	 * Return TRUE, not false: returning false tells lruvec_add_folio() to
	 * run its LEGACY fallback (update_lru_size(+nr) + list_add onto a real
	 * lruvec->lists[lru]) on a folio that is STILL TRACKED and that Marie
	 * never credited to mz->lru_zone_size. That stray, never-debited mz
	 * credit + a folio cross-linked onto a legacy list is exactly the
	 * mz->lru_zone_size underflow ("lru_size -1") we were chasing. TRUE
	 * means "Marie owns it, do not add anywhere" -- which is what "retain
	 * ownership" requires.
	 */
	pfn = folio_pfn(folio);
	if (pfn < marie_state_size &&
	    (READ_ONCE(marie_state[pfn]) & MARIE_PFN_TRACKED))
		return true;

	/*
	 * Fresh installs always land on the head gen regardless of any
	 * workingset/active signal (tier's former role encoding that
	 * distinction is retired -- see state.h's byte-layout block): a
	 * folio being installed for the first time is already at the
	 * youngest possible position, so there is nothing left for a
	 * hotness signal to express here. PG_active is still normalised to
	 * 0 before computing inst_lru, matching marie_evict_locked and the
	 * putback path's own PG_active hygiene. PG_workingset is left
	 * untouched: workingset_eviction's shadow encoding needs it at the
	 * next eviction.
	 */
	was_active = folio_test_active(folio);
	if (was_active)
		folio_clear_active(folio);

	/*
	 * folio->lru MUST be re-initialised here. A recycled folio arrives
	 * with LIST_POISON{1,2} from the prior owner's list_del, and the
	 * eventual marie_evict_locked's list_del_init would walk the
	 * poison pointers and fault.
	 */
	INIT_LIST_HEAD(&folio->lru);

	type = folio_is_file_lru(folio);
	zone = folio_zonenum(folio);
	/* Install into THIS lruvec's own youngest (head) gen. */
	head = (u8)atomic_read(&marie_head_gen[type]);

	/*
	 * Publish per-PFN state byte + scan bitmap + the global
	 * gen_occupied++. See pfn_install.h::marie_pfn_publish_inherit.
	 * @old_b is the replaced byte; the counter credit is derived from the
	 * (old_b -> published) transition at the tail of this function rather
	 * than stated here. See account.h.
	 */
	new_b = marie_pfn_publish_inherit(folio, type, head, zone, &old_b);
	/*
	 * Install-cadence aging (global). Count installs onto the head gen and
	 * seal the generation once it has accumulated marie_gen_growth_live[type]
	 * pages, advancing the head so subsequent installs land in a fresh
	 * younger gen. This stratifies folios by age PROACTIVELY so the oldest
	 * gen holds genuinely-old folios and reclaim does not waste scans
	 * rotating hot ones. Counter is global/atomic (installs run under
	 * different per-lruvec lru_locks); reset only on a real advance, so a
	 * blocked attempt (next slot still draining) retries on the next install.
	 *
	 * Count PAGES, not folios: a large folio (THP) deposits folio_nr_pages
	 * pages onto the head gen in one install, so the head must advance in
	 * proportion. Incrementing by 1 per folio advanced the head up to
	 * 512x too slowly under THP=always -- the entire anon set piled into the
	 * head gen, reclaim (which scans the aged gens) found nothing to isolate,
	 * and the node went all_unreclaimable with free swap: premature OOM and a
	 * multi-CPU lru_lock livelock (tens-of-seconds GUI freeze). order-0 is
	 * unaffected (folio_nr_pages == 1).
	 */
	if (atomic_long_add_return(folio_nr_pages(folio),
				   &marie_gen_installs[type]) >=
	    marie_gen_growth_threshold(type)) {
		if (marie_try_advance_head_mlv(type))
			atomic_long_set(&marie_gen_installs[type], 0);
	}

	/*
	 * Atomic PG_active->0 + PG_lru->1 in one mask write. PG_active was
	 * cleared above when set; the mask write keeps the invariant
	 * against the defensive case where another path set PG_active
	 * between then and now. Ordered AFTER the state-byte publish so a
	 * concurrent __page_cache_release observing PG_lru=1 also observes
	 * marie_state[pfn] & MARIE_PFN_TRACKED.
	 */
	set_mask_bits(&MARIE_FOLIO_FLAGS(folio), BIT(PG_active), BIT(PG_lru));

	/*
	 * Credit is DERIVED from the publish transition, not stated: the byte
	 * this install replaced was not TRACKED (the early-out above proved
	 * it), so pred goes 0 -> 1 and marie_acct_settle_locked applies
	 * exactly +1 / +nr. The (lru, zone) bucket comes from the published
	 * byte, so the eventual debit -- wherever it happens -- names the same
	 * slot. See account.h.
	 */
	marie_acct_settle_locked(lv, folio_nr_pages(folio), old_b, new_b);

	return true;
}

bool marie_evict_locked(struct folio *folio)
{
	struct lruvec *lv = folio_lruvec(folio);

	lockdep_assert_held(&lv->lru_lock);
	lockdep_assert_irqs_disabled();

	/*
	 * folio->lru is either a self-loop (install/flush leave it that
	 * way, and the per-PFN paradigm never re-attaches it onto a
	 * Marie-owned list) or on legacy lruvec->lists[lru] after a
	 * drain handed it off. list_del_init is a no-op in the first
	 * case and a legacy-list removal in the second; the caller
	 * holds lruvec->lru_lock for the latter, so no extra Marie-side
	 * lock is required.
	 */
	list_del_init(&folio->lru);

	/*
	 * Drops PG_active for shrink_folio_list, which trips
	 * VM_BUG_ON_FOLIO(folio_test_active) otherwise.
	 *
	 * This no longer has any accounting significance: the debit's
	 * (lru, zone) bucket is reconstructed from the state byte inside
	 * marie_state_drop_pfn -> marie_acct_settle_locked, never from
	 * folio_lru_list(), so a PG_active that some path re-stamped between
	 * install and here can no longer misdirect the debit into ACTIVE_*
	 * (an index Marie's install never credited). See account.h.
	 */
	if (folio_test_active(folio))
		folio_clear_active(folio);

	/*
	 * Clear PG_lru BEFORE marie_state_drop_pfn so a concurrent
	 * del-side path gated on folio_test_clear_lru cannot observe
	 * (state=TRACKED, PG_lru=1) -> Marie del again recursion.
	 * drop_pfn then wipes the per-PFN state (byte, bitmap,
	 * l2_range_count, memcg L1) which is the only Marie tracking
	 * for this folio.
	 *
	 * Idempotent for callers that already cleared PG_lru via
	 * folio_test_clear_lru before reaching evict
	 * (__page_cache_release, marie_state_shrink_lruvec claim loop).
	 */
	folio_clear_lru(folio);
	marie_state_drop_pfn(folio);

	return true;
}

/*
 * marie_evict_counters_only - reclaim-isolate per-folio counter decrement
 * that also sets marie_state[]'s ISOLATED bit, but PRESERVES its TRACKED
 * bit.
 *
 * The per-PFN state byte staying TRACKED throughout shrink_folio_list is
 * the race defence: marie_folio_install's "already TRACKED" early-out
 * makes a concurrent install on this PFN bail, so install cannot set
 * PG_lru on the folio while shrink_folio_list is reclaiming it. (The
 * earlier full marie_evict_isolated cleared TRACKED inline; a concurrent
 * install would then succeed, set PG_lru, and trip
 * PAGE_FLAGS_CHECK_AT_FREE at free_unref_folios in the success path.)
 *
 * ISOLATED is set here, at isolate, as early as possible (see the body):
 * this is the gate (state.h) that stops every other GEN-mutator (walker
 * promote-on-access, MADV_COLD, MADV_FREE, defrag restamp) from touching
 * this pfn's GEN until putback releases it. GEN now lives in the same
 * byte as ISOLATED, so setting ISOLATED is a proper cmpxchg retry loop,
 * not a plain OR-in-place store: a concurrent promote-on-access racing
 * this exact byte will have its own cmpxchg fail and retry once ISOLATED
 * lands, observing it and declining -- see marie_state_move_to_gen's
 * comment for why this closes the residual race the prior marie_state +
 * marie_age split could not. @cur, captured at the instant the CAS
 * lands, is the authoritative pre-isolate (gen, type) coordinate: the
 * bitmap bit + gen_occupied for it are retired immediately below, using
 * that exact snapshot rather than a later re-read (nothing else can
 * change it once ISOLATED is set, so the two are equivalent, but the
 * snapshot avoids a redundant load).
 *
 * Caller-side gates that hold throughout this path:
 *   1. folio_try_get()        - reference held, folio cannot be freed.
 *   2. folio_test_clear_lru() - PG_lru cleared atomically, gating
 *                                external del paths.
 *   3. install_local TRACKED early-out (above)
 *
 * Counters are decremented immediately so the in-flight folio does not
 * inflate lruvec_lru_size() and skew reclaim pressure heuristics during
 * shrink_folio_list. The (type, zone, gen) bitmap bit + gen_occupied are
 * torn down HERE so the in-flight folio leaves the scan's candidate
 * index immediately -- no other scanner can re-find it (unlike the
 * retired coarse-hint design, this bitmap is exact); only the
 * TRACKED/ISOLATED bits are deferred (to the buddy free hook for
 * reclaimed folios). Survivors go through the putback path, which
 * publishes a fresh marie_state[] value and clears ISOLATED.
 */
void marie_evict_counters_only(struct folio *folio)
{
	struct lruvec *lv = folio_lruvec(folio);
	int zone = folio_zonenum(folio);
	unsigned long pfn = folio_pfn(folio);
	u8 cur = 0;
	bool tracked = false;

	/*
	 * Set ISOLATED first, before any other work in this function -- the
	 * narrower the gap after the caller's folio_test_clear_lru() claim,
	 * the smaller the window other GEN-mutators have to race (see the
	 * function comment above).
	 */
	if (pfn < marie_state_size) {
		cur = READ_ONCE(marie_state[pfn]);
		while (cur & MARIE_PFN_TRACKED) {
			if (try_cmpxchg(&marie_state[pfn], &cur,
					 cur | MARIE_PFN_ISOLATED)) {
				tracked = true;
				break;
			}
		}
	}

	if (unlikely(!list_empty(&folio->lru))) {
		/*
		 * Defensive: an mm/swap.c batch path lacking a Marie gate
		 * may have placed this folio onto a legacy lruvec list via
		 * lruvec_add_folio_tail. The caller's list_add(&f->lru, ...)
		 * would then corrupt that list. Detach under lru_lock first;
		 * DO NOT fall back to lru_marie_del_folio (it would clear
		 * TRACKED via marie_state_drop_pfn, breaking the deferred-
		 * teardown invariant the putback path relies on).
		 */
		VM_WARN_ON_ONCE_FOLIO(1, folio);
		scoped_guard(spinlock_irq, &lv->lru_lock)
			list_del_init(&folio->lru);
	}

	if (folio_test_active(folio))
		folio_clear_active(folio);

	/*
	 * The ISOLATED CAS above IS the accounting transition: pred goes
	 * 1 -> 0, so marie_acct_settle_isolate derives exactly one debit from
	 * it. Nothing states a -1 independently of the state change, which is
	 * what makes a second debit for this folio unrepresentable -- any
	 * later teardown path observes pred(old) == 0 and computes 0.
	 *
	 * @cur is the pre-CAS byte and @cur|ISOLATED is what landed; if the
	 * folio was not TRACKED at all, both sides have pred 0 and settle is a
	 * no-op. settle owns the local_irq_save/restore this lock-free path
	 * needs against same-CPU softirq reentrancy on fbc->lock and the
	 * per-CPU vmstat diff (see account.h, and 9c6a93782's lockup history).
	 */
	{
		u8 new_b = tracked ? (u8)(cur | MARIE_PFN_ISOLATED) : cur;

		marie_acct_settle_isolate(lv, folio_nr_pages(folio), cur, new_b);
		marie_gen_occ_settle(pfn, cur, new_b);
	}

	/*
	 * Retire the scan-index bit for the pre-isolate coordinate. Occupancy
	 * was settled from the byte transition above, so this return value is
	 * advisory.
	 */
	if (tracked) {
		u8 gen = (cur & MARIE_PFN_GEN_MASK) >> MARIE_PFN_GEN_SHIFT;
		u8 tb = (cur & MARIE_PFN_TYPE_MASK) ? 1 : 0;

		marie_bm_retire(pfn, tb, zone, gen);
	}
}

/*
 * Promotes the folio straight to the head gen; marie_state_inc_tier
 * delegates to marie_state_move_to_gen internally.
 */
void lru_marie_mark_accessed(struct folio *folio)
{
	unsigned long pfn = folio_pfn(folio);
	u8 state;

	if (!lru_marie_enabled() || !marie_state_ready())
		return;
	if (pfn >= marie_state_size)
		return;
	state = READ_ONCE(marie_state[pfn]);
	if (!(state & MARIE_PFN_TRACKED))
		return;

	/* Promote-on-access. Lock-free fault/hit path. */
	marie_state_inc_tier(pfn);
	/* Mark the page as recently accessed for the workingset estimator. */
	if (folio_test_clear_referenced(folio))
		folio_set_workingset(folio);
}
EXPORT_SYMBOL_GPL(lru_marie_mark_accessed);

/*
 * Per-cpu folio_batch LRU-op hooks (declared in <linux/lru_marie.h>).
 * Each applies the op directly on the folio's per-PFN state and returns
 * true so mm/swap.c skips the legacy folio_batch; false (Marie off / folio
 * untracked) falls through to the legacy path. All run lock-free:
 * marie_state_move_to_gen is CAS-based (and checks marie_state[]'s
 * ISOLATED gate), matching the no-lru_lock contract of these entry points.
 */

/*
 * Demote: relocate to the oldest live gen so Marie's next scan reclaims
 * it promptly. Used for the EXPLICIT user "make cold" madvise (MADV_COLD
 * -> folio_deactivate / deactivate_file_folio). Reclaim-internal hints
 * (activate / rotate) deliberately do NOT demote -- see those hooks.
 */
static bool marie_folio_demote(struct folio *folio)
{
	int type, oldest;

	if (!lru_marie_enabled() || !folio_marie_test_tracked(folio))
		return false;
	type = folio_is_file_lru(folio);
	oldest = marie_find_oldest_occupied_mlv(type);
	if (oldest >= 0)
		marie_state_move_to_gen(folio_pfn(folio), (u8)oldest);
	return true;
}

bool lru_marie_deactivate(struct folio *folio)
{
	return marie_folio_demote(folio);
}
EXPORT_SYMBOL_GPL(lru_marie_deactivate);

/*
 * rotate: NO-OP for Marie folios (skip the legacy batch). Like activate
 * this is a reclaim-internal hint (folio_rotate_reclaimable fires on
 * writeback completion of a PG_reclaim folio). An actively reclaimed Marie
 * folio is isolated (PG_lru cleared) so this is rarely reached, and Marie's
 * gen aging already orders reclaim -- no per-PFN state change is wanted.
 */
bool lru_marie_rotate(struct folio *folio)
{
	return lru_marie_enabled() && folio_marie_test_tracked(folio);
}
EXPORT_SYMBOL_GPL(lru_marie_rotate);

/*
 * activate: NO-OP for Marie folios (but skip the legacy batch by returning
 * true). folio_activate is driven mostly by shrink_folio_list's
 * FOLIOREF_ACTIVATE during reclaim, and Marie already decides retention
 * there via its tier vote in folio_check_references. Promoting to the head
 * gen on top would pull referenced folios out of the oldest gen on every
 * reclaim pass; under an all-hot workload that starves reclaim entirely
 * (OOM with GBs of unreclaimable inactive_anon). The explicit-access
 * channel is folio_mark_accessed -> lru_marie_mark_accessed (tier bump),
 * which must not be double-counted here.
 */
bool lru_marie_activate(struct folio *folio)
{
	return lru_marie_enabled() && folio_marie_test_tracked(folio);
}
EXPORT_SYMBOL_GPL(lru_marie_activate);

/*
 * MADV_FREE: make the anon folio reclaim-without-writeback. Clear the
 * dirtiness signals synchronously (what the legacy lru_lazyfree move_fn
 * does) and demote so Marie frees it promptly without swap on the next
 * scan. type is read before clearing swapbacked (folio_is_file_lru flips
 * once swapbacked is gone); the Marie byte keeps its anon TYPE, so demote
 * stays within the anon gen ring.
 */
bool lru_marie_lazyfree(struct folio *folio)
{
	int type, oldest;

	if (!lru_marie_enabled() || !folio_marie_test_tracked(folio))
		return false;
	type = folio_is_file_lru(folio);
	folio_clear_active(folio);
	folio_clear_referenced(folio);
	folio_clear_swapbacked(folio);
	count_vm_events(PGLAZYFREE, folio_nr_pages(folio));
	oldest = marie_find_oldest_occupied_mlv(type);
	if (oldest >= 0)
		marie_state_move_to_gen(folio_pfn(folio), (u8)oldest);
	return true;
}
EXPORT_SYMBOL_GPL(lru_marie_lazyfree);

/*
 * lru_marie_test_tracked (public API in <linux/lru_marie.h>).
 */
bool lru_marie_test_tracked(const struct folio *folio)
{
	return folio_marie_test_tracked(folio);
}
EXPORT_SYMBOL_GPL(lru_marie_test_tracked);

/*
 * lru_marie_free_page_hook (public API in <linux/lru_marie.h>).
 * Thin wrapper over marie_state_drop_pfn_at_free so the page allocator
 * can call the hook without including the private state.h.
 */
void lru_marie_free_page_hook(unsigned long pfn)
{
	marie_state_drop_pfn_at_free(pfn);
}
EXPORT_SYMBOL_GPL(lru_marie_free_page_hook);

/*
 * marie_del_folio_locked - lru_marie_del_folio body.
 *
 * External-removal entry: if the folio is still Marie-tracked, do the
 * full evict via marie_evict_locked, which routes through
 * marie_acct_settle_locked and owns the ENTIRE counter wind-down -- including
 * the single marie_nr_folios -1. The caller does no accounting of its
 * own; an earlier caller-side -1 predated the account.h funnel and
 * double-counted marie_nr_folios on every generic del of a tracked folio.
 *
 * Lock contract: caller holds lruvec->lru_lock. No Marie lock is taken
 * here -- the lru_lock invariant already serialises every Marie state
 * mutation. See the comment above the call site in lru_marie_del_folio
 * for the full protection-model rationale.
 *
 * Returning true tells the dispatcher (lruvec_del_folio in
 * include/linux/mm_inline.h) "Marie owns this folio, do not fall
 * through to legacy".
 *
 * The not-tracked branch returns true defensively. Under the lru_lock
 * invariant it is unreachable -- the caller's TRACKED fast-path test
 * already gated entry here -- but returning true keeps the safe
 * behaviour if the invariant ever regresses: a stray legacy
 * update_lru_size on a folio Marie already accounted would double-
 * decrement mz->lru_zone_size.
 */
bool marie_del_folio_locked(struct folio *folio)
{
	lockdep_assert_held(&folio_lruvec(folio)->lru_lock);
	lockdep_assert_irqs_disabled();

	if (!folio_marie_test_tracked(folio))
		return true;
	return marie_evict_locked(folio);
}

/*
 * ---------------------------------------------------------------------
 *  global init
 * ---------------------------------------------------------------------
 */

/*
 * Global per-type locks.  Marie is desktop/global-only: there is a single
 * aging clock and a single global track bitmap, so the per-type serialising
 * lock is one global instance per type (anon / file), not one per lruvec.
 * The only holders are the THP-install and split-tail paths in core.c.
 */
struct marie_type marie_type_locks[ANON_AND_FILE];

/* marie_type_init: caller-side scalar/lock initialisation only. */
static void marie_type_init(struct marie_type *t, int type)
{
	spin_lock_init(&t->type_lock);
	t->type = type;
}

int marie_counters_init(void)
{
	int t;

	for (t = 0; t < ANON_AND_FILE; t++)
		marie_type_init(&marie_type_locks[t], t);

	return percpu_counter_init(&marie_nr_folios, 0, GFP_KERNEL);
}
