/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_LRU_MARIE_H
#define _LINUX_LRU_MARIE_H

/*
 * Marie LRU — public API.
 *
 * Marie represents each folio's reclaim state as a single byte in a
 * flat per-PFN array allocated once at boot (see mm/lru_marie/state.h
 * for the byte layout). install / delete / aging are single byte
 * writes — no allocation in any fault-path operation, no per-folio
 * linked-list linkage. isolate batches the array with a persistent
 * cursor and SIMD scan.
 *
 * The vmscan core invokes Marie only when lru_marie_enabled() is true;
 * otherwise the in-tree reclaim paths run unchanged.
 *
 * Selected once at boot: the compile-time default comes from
 * CONFIG_LRU_MARIE_DEFAULT_ON, and either default can be overridden with
 * `lru_marie=0` / `lru_marie=1` on the kernel command line. There is no
 * runtime toggle -- the static key is set before anything is tracked, and
 * /sys/kernel/mm/lru_marie/enabled is read-only.
 *
 * This header exposes only the thin dispatch surface that mm/vmscan.c,
 * mm/swap.c, mm/memcontrol.c, etc. need to know about. Everything
 * else lives inside mm/lru_marie/.
 */

#include <linux/atomic.h>
#include <linux/jump_label.h>
#include <linux/mm_types.h>
#include <linux/mmzone.h>
#include <linux/types.h>

struct folio;
struct lruvec;
struct mem_cgroup;
struct page_vma_mapped_walk;
struct pglist_data;
struct scan_control;

#ifdef CONFIG_LRU_MARIE

#ifdef CONFIG_LRU_MARIE_DEFAULT_ON
DECLARE_STATIC_KEY_TRUE(lru_marie_enabled_key);
#else
DECLARE_STATIC_KEY_FALSE(lru_marie_enabled_key);
#endif

/**
 * lru_marie_enabled - is the Marie reclaim path currently active?
 *
 * Inlined static-branch check. Default comes from
 * CONFIG_LRU_MARIE_DEFAULT_ON; when on, the static-key compiles into a
 * single unconditional jump that the predictor resolves in zero
 * cycles. The MGLRU/Legacy paths cost nil when Marie is enabled (the
 * common case) since the branch falls through to the Marie-side code
 * without any conditional dispatch overhead.
 */
static inline bool lru_marie_enabled(void)
{
	return static_branch_likely(&lru_marie_enabled_key);
}

DECLARE_STATIC_KEY_FALSE(marie_state_ready_key);

/**
 * marie_state_ready - has the per-PFN marie_state[] array been allocated?
 *
 * Distinct from lru_marie_enabled(): the enable key is the boot-set
 * reclaim-policy static branch, whereas this key latches true once
 * marie_state[] is allocated at init and NEVER flips back -- the array is
 * never freed for the kernel's lifetime.
 *
 * The page-free hook (lru_marie_free_page_hook) must gate on THIS key, not
 * on lru_marie_enabled(): the per-PFN array is allocated at subsys_initcall,
 * so early in boot lru_marie_enabled() can already be true (boot param set)
 * while marie_state[] is still NULL. Gating the free hook on array-readiness
 * keeps it from dereferencing a not-yet-allocated array, while still wiping
 * stale TRACKED bits at the buddy handoff for every page once the array
 * exists.
 */
static inline bool marie_state_ready(void)
{
	return static_branch_unlikely(&marie_state_ready_key);
}

/**
 * lru_marie_mark_accessed - Marie's hot-signal entry point for folio_mark_accessed.
 *
 * Promotes @folio straight to the head gen in the per-PFN marie_state[]
 * byte (marie_state_inc_tier -> marie_state_move_to_gen). The walker
 * promotes on young-bit hits via the same channel, and this helper lets
 * external "user just touched" callers (folio_mark_accessed) feed it
 * too. Calling this from the user access hot path costs at most one CAS;
 * no slab alloc, no enqueue.
 *
 * PG_referenced is the first rung of the ladder rather than a parallel
 * signal: folio_mark_accessed() sets it on a first access and only calls
 * this on a second one, and this clears it again (handing it to
 * PG_workingset for the refault estimator). So the two never both stand
 * for "hot" at once, which is what the earlier double-counting objection
 * was about -- back when the alternative reconciliation rule (treat any
 * of {PG_referenced, PG_active} as promote-in-place during reclaim)
 * starved kswapd under fault-burst workloads.
 */
void lru_marie_mark_accessed(struct folio *folio);

/**
 * lru_marie_test_tracked - is @folio currently tracked by Marie?
 *
 * Reads the per-PFN marie_state[folio_pfn(folio)] byte's TRACKED bit.
 * Returns false if Marie is disabled, the PFN is out of range, or the
 * folio is not Marie-tracked.
 *
 * Used by mm/swap.c per-cpu folio_batch entry points (rotate / activate
 * / deactivate / lazyfree) to skip queueing Marie folios: those paths
 * do legacy lruvec_del_folio + lruvec_add_folio, whose list_del/list_add
 * assume the folio is on a legacy lruvec list. Marie folios sit on a
 * self-loop (folio->lru points at itself), not on a legacy list, so a
 * legacy del/add would corrupt the list. (mz->lru_zone_size is balanced
 * for Marie folios now -- marie_update_lru_size credits it at install --
 * so the hazard is list corruption, not count underflow.)
 */
bool lru_marie_test_tracked(const struct folio *folio);

/*
 * Per-cpu folio_batch LRU-op interface. mm/swap.c's folio_activate /
 * folio_deactivate / deactivate_file_folio / folio_rotate_reclaimable /
 * folio_mark_lazyfree each call the matching hook below; a true return
 * means Marie owns the folio and has applied the operation directly on
 * its per-PFN state, so the caller must NOT queue the folio onto the
 * legacy per-cpu folio_batch (which assumes legacy-LRU list/mz invariants
 * Marie folios break). A false return (Marie disabled or folio untracked)
 * lets the caller fall through to the legacy folio_batch path unchanged.
 *
 * This mirrors lru_marie_add_folio / lru_marie_del_folio's bool contract:
 * the Marie-specific semantics live here in mm/lru_marie/, not as inline
 * gates scattered across mm/swap.c.
 *
 * Marie-state equivalents:
 *   deactivate / _file -> demote  (move to oldest gen, tier 0) [MADV_COLD]
 *   lazyfree           -> clear swapbacked + demote            [MADV_FREE]
 *   activate / rotate  -> no-op (skip the batch only)
 *
 * activate / rotate are reclaim-internal hints: Marie already decides
 * hotness via its tier vote in folio_check_references and orders reclaim
 * by gen aging, so promoting/rotating here would only fight reclaim (an
 * activate-promote starves it under all-hot workloads). Only the explicit
 * user madvise paths (deactivate=MADV_COLD, lazyfree=MADV_FREE) map to a
 * real Marie-state change.
 */
bool lru_marie_activate(struct folio *folio);
bool lru_marie_deactivate(struct folio *folio);
bool lru_marie_rotate(struct folio *folio);
bool lru_marie_lazyfree(struct folio *folio);

/**
 * lru_marie_free_page_hook - canonical per-PFN state teardown at buddy
 *                            handoff.
 *
 * Invoked from mm/page_alloc.c::free_pages_prepare for every page about
 * to enter the buddy allocator. When marie_state[pfn] still carries
 * TRACKED -- which happens whenever the reclaim isolate path
 * (marie_evict_counters_only) decremented counters but intentionally
 * preserved the state byte so install_local's TRACKED early-out kept
 * blocking concurrent installs during shrink_folio_list -- this wipes
 * the byte, the global (type, gen, tier) bitmap bit, and the
 * gen_occupied slot in one lock-free pass.
 *
 * After this hook the next install at the same PFN starts from a clean
 * state byte regardless of how quickly the page is re-allocated; no
 * deferred drop pass is needed at the reclaim caller side.
 *
 * Static-branch gated by lru_marie_enabled() at the call site to keep
 * the !Marie build / runtime byte-identical.
 */
void lru_marie_free_page_hook(unsigned long pfn);

/**
 * lru_marie_uncharge_backstop - settle an escaped tracked folio's counters
 *                               at the last point its memcg/lruvec resolves.
 *
 * Called from mm/memcontrol.c::uncharge_folio just before folio->memcg_data
 * is zeroed. A folio that reaches free without going through Marie's evict
 * (its scan bit still set) left its install +nr un-debited; the page-free
 * hook cannot undo it (memcg_data is already gone there). This is that
 * universal confluence point -- every charged LRU folio is uncharged before
 * buddy handoff -- so the global debit (marie_nr_folios + the lruvec's vmstat
 * lru_size, resolved via the still-live memcg) is unified here. No-op unless
 * the per-PFN scan bit is still set (the exactly-once token cleared by the
 * normal evict path).
 */
void lru_marie_uncharge_backstop(struct folio *folio, struct mem_cgroup *memcg);

/*
 * Return value of lru_marie_shrink_lruvec(): a mask of the LRU type(s) the
 * Marie pick driver actually scanned this call. shrink_lruvec's legacy orphan
 * drain reclaims ONLY these types (it zeroes the nr[] of any unset type), so
 * it never cuts a type Marie's swappiness / clean_min_ratio / ANON_STRICT
 * policy protected -- unlike stock get_scan_count, which the legacy drain
 * would otherwise follow blindly (SCAN_EQUAL at priority 0, etc.).
 */
#define MARIE_DRAIN_ANON	0x1u
#define MARIE_DRAIN_FILE	0x2u

/**
 * lru_marie_shrink_lruvec - Marie's replacement for shrink_lruvec().
 *
 * Called from mm/vmscan.c shrink_lruvec() when lru_marie_enabled() is true.
 * Updates sc->nr_reclaimed in place. Returns a MARIE_DRAIN_* mask of the
 * type(s) it scanned so the caller's legacy orphan drain can mirror the pick
 * policy instead of running stock get_scan_count's.
 */
unsigned int lru_marie_shrink_lruvec(struct lruvec *lruvec, struct scan_control *sc);

/**
 * lru_marie_add_folio - try to register @folio with Marie.
 * @lruvec:     the lruvec @folio is being added to
 * @folio:      the folio
 * @reclaiming: caller hint, unused at this stage
 *
 * Returns true if Marie took ownership of the folio (and the caller must
 * skip the legacy lruvec list_add) — false if Marie declined (gate off,
 * lruvec state unavailable, or allocation failed) and the caller should
 * fall through to the existing MGLRU / Legacy path.
 */
bool lru_marie_add_folio(struct lruvec *lruvec, struct folio *folio, bool reclaiming);

/**
 * lru_marie_orphan_add - pure legacy LRU add for an untracked orphan.
 * @lruvec: the lruvec @folio is being added to
 * @folio:  an untracked (non-Marie) folio
 * @tail:   add to the list tail rather than the head
 *
 * A del+add move_fn (swap.c: lru_activate / lru_deactivate{,_file} /
 * lru_lazyfree) and the legacy reclaim putback (vmscan.c:
 * move_folios_to_lru) run lruvec_del_folio() -- legacy del, mz -nr for an
 * untracked folio -- and then add the folio back. Routing that add through
 * lruvec_add_folio() -> lru_marie_add_folio() would ADOPT the orphan into
 * Marie; the install credits Marie's own bucket but the original -nr was a
 * legacy debit, so the legacy mz->lru_zone_size drifts and a later del
 * underflows ("marie underflow-del"). This does the +nr legacy leg only,
 * never adopting. Callers MUST first bail on lru_marie_test_tracked()
 * folios -- a tracked folio is Marie-owned and must never touch a legacy
 * list.
 */
void lru_marie_orphan_add(struct lruvec *lruvec, struct folio *folio, bool tail);


/**
 * lru_marie_del_folio - try to remove @folio from Marie.
 * @lruvec:     the lruvec @folio is being removed from
 * @folio:      the folio
 * @reclaiming: caller hint, unused at this stage
 *
 * Returns true iff @folio was tracked by Marie and has now been removed.
 * Returns false if @folio was on the legacy LRU instead, in which case
 * the caller continues with the legacy delete path.
 */
bool lru_marie_del_folio(struct lruvec *lruvec, struct folio *folio, bool reclaiming);

/**
 * lru_marie_release_folio - outer-level release for a TRACKED folio
 *  reaching refcount 0 (called from __page_cache_release).
 * @folio:    the folio being released
 * @lruvecp:  caller's lruvec batch pointer (may be NULL or hold a lock)
 * @flagsp:   caller's irqsave flags slot
 *
 * The dispatch contract: when Marie is enabled, upstream callers MUST
 * gate by folio_marie_test_tracked() and call this helper for TRACKED
 * folios INSTEAD OF the legacy folio_test_lru / lruvec_del_folio path.
 * A TRACKED folio sits on a Marie self-loop, not on a legacy lruvec
 * list, so legacy lruvec_del_folio's list_del would corrupt it; this
 * helper unlinks the self-loop and debits mz instead. TRACKED is the
 * single source of truth.
 *
 * Internally: relocks @lruvecp to @folio's lruvec with IRQs disabled,
 * re-tests TRACKED under the lock, runs Marie's del (which leaves mz
 * untouched), and clears PG_lru. If TRACKED was cleared between the
 * caller's outer test and our lock acquisition (race with drain or
 * evict), falls back to the legacy del so accounting stays coherent.
 * Leaves the lock held in *@lruvecp for the caller's batch context.
 */
void lru_marie_release_folio(struct folio *folio, struct lruvec **lruvecp,
			     unsigned long *flagsp);

/**
 * lru_marie_split_folio - install a freshly-split tail folio under Marie.
 * @lruvec:     head folio's lruvec (caller holds lru_lock)
 * @head:       THP head folio currently tracked by Marie
 * @new_folio:  tail folio created by __split_huge_page
 *
 * Mirror of mm/huge_memory.c::lru_add_split_folio's
 * "list_add_tail(&new_folio->lru, &folio->lru)" for the case where
 * @head is Marie-tracked. Publishes the per-PFN state byte for the new
 * folio so the dispatcher routes its eventual del through Marie;
 * otherwise the new tail would be untracked, dispatcher del would fall
 * to legacy update_lru_size, mlv->types[].nr_pages would not decrement,
 * and reclaim heuristics would drift.
 *
 * Caller MUST verify lru_marie_enabled() && folio_marie_test_tracked(head)
 * before calling. Caller holds @lruvec->lru_lock; the per-type lock is
 * taken internally.
 *
 * Caller is responsible for folio_set_lru(@new_folio) AFTER this
 * returns — the "state byte published before PG_lru" rule is preserved
 * by the call ordering.
 */
void lru_marie_split_folio(struct lruvec *lruvec, struct folio *head,
			 struct folio *new_folio);

/**
 * lru_marie_split_folio_isolated - register a reclaim-split tail under Marie.
 * @head:       THP head folio, Marie-tracked but OFF the LRU (reclaim-isolated)
 * @new_folio:  tail folio created by __split_huge_page during reclaim
 *
 * The reclaim-split counterpart to lru_marie_split_folio. When page reclaim
 * splits a huge page, lru_add_split_folio() runs with a non-NULL @list (the
 * head is frozen OFF the LRU, PG_lru clear) and threads each tail onto that
 * reclaim list instead of the LRU. Without this, the tail stays UNTRACKED:
 * Marie's reclaim/putback machinery (which assumes every folio it handles is
 * TRACKED) then either mis-accounts it or, once a generic putback re-sets
 * PG_lru, produces a ¬TRACKED∧PG_lru "escapee" whose eventual legacy del
 * underflows mem_cgroup mz->lru_zone_size and corrupts memory.
 *
 * Publishes the tail's per-PFN TRACKED state at @head's gen (tier 0) and bumps
 * marie_nr_folios, mirroring lru_marie_split_folio -- but does NOT set PG_lru
 * and does NOT touch folio->lru linkage: the caller threads the tail onto the
 * reclaim @list, and the tail stays off-LRU exactly like the isolated head.
 * No lru_size credit: the head's install +N already covered the compound and
 * the reclaim isolate debited it; each tail's own del/putback balances from
 * there. No-op when @head is not Marie-tracked.
 *
 * Caller holds @head's lruvec->lru_lock (lru_add_split_folio asserts it).
 */
void lru_marie_split_folio_isolated(struct folio *head,
				    struct folio *new_folio);

/**
 * lru_marie_look_around - opportunistic PMD scan during rmap reference check.
 * @pvmw: page-vma-mapped walk supplied by folio_referenced_one()
 * @nr:   number of consecutive PTEs of the target folio at pvmw->address
 *
 * Called from rmap.c::folio_referenced_one() in the Marie branch with
 * pvmw->ptl already held.  Clears the target folio's young bit (returning
 * its previous state) and, while the PTL is hot, scans up to
 * MARIE_LOOK_AROUND_BATCH PTEs of the surrounding PMD, clearing young bits
 * found there too.  This batches what would otherwise be one rmap walk
 * per neighbouring folio and improves the accuracy of subsequent
 * folio_referenced() calls.  Returns true iff the target's own PTE(s)
 * were young.
 */
bool lru_marie_look_around(struct page_vma_mapped_walk *pvmw, unsigned int nr);

/**
 * lru_marie_age_node - kswapd pre-reclaim aging hook for Marie.
 * @pgdat: kswapd's pgdat
 * @sc:    kswapd's scan_control
 *
 * Called from kswapd_age_node() when Marie owns the LRU.  Drives the
 * proactive PTE walker (marie_walk_pgdat internally) so per-PFN tier
 * encoding has accurate hot/cold ordering by the time direct reclaim
 * picks the oldest gen. Internally rate-limited; safe to call from any
 * kswapd cycle.
 */
void lru_marie_age_node(struct pglist_data *pgdat, struct scan_control *sc);

#ifdef CONFIG_LRU_MARIE_DEFRAG
/**
 * lru_marie_defrag_active - is the lru_marie/defrag switch on, i.e. should
 * Marie defrag replace stock compaction in kcompactd? Gates the swap in BOTH
 * of mm/compaction.c's kcompactd paths (true => call lru_marie_defrag_pgdat
 * instead of the stock compaction).
 */
bool lru_marie_defrag_active(void);

/**
 * lru_marie_defrag_pgdat - run one Marie defrag pass for @pgdat.
 * @may_drop: urgency from the calling path -- true on the demand path
 *            (kcompactd_do_work; a high-order allocation is blocked) so Marie
 *            MOVEs + DROPs cold-dead clean file; false on the proactive
 *            background path so Marie MOVEs only.
 *
 * Called from mm/compaction.c INSTEAD OF the stock compaction (compact_node on
 * the proactive path, the demand sweep on kcompactd_do_work) when
 * lru_marie_defrag_active() is true. Runs in the kcompactd kthread context
 * (may sleep). See mm/lru_marie/defrag.c.
 */
void lru_marie_defrag_pgdat(struct pglist_data *pgdat, bool may_drop);
#endif

/**
 * lru_marie_swappiness_changed - notify Marie that a swappiness value
 *                                has been written via sysctl or memcg.
 *
 * Resets the single global swap_bias counter to zero so the next reclaim
 * cycle starts from a neutral state under the new swappiness. Stale bias
 * accumulated under the previous value would otherwise steer the first
 * several picks in the wrong direction, especially across transitions into
 * or out of the special-value range {0, 1, MAX_SWAPPINESS}.
 *
 * Desktop/global-only: the controller's only state is one node-wide
 * atomic64, so this is a single atomic64_set -- no per-lruvec walk.
 *
 * Safe to call from sysctl proc_handler context.
 */
void lru_marie_swappiness_changed(void);

/*
 * Runtime-tunable knobs exposed via /sys/kernel/mm/lru_marie/.
 * Read with READ_ONCE; sysfs store writes with WRITE_ONCE. Hot-path
 * snapshots are taken at the top of each loop iteration so a concurrent
 * write only takes effect on the next pass.
 */
extern unsigned long marie_walker_interval_critical;	/* jiffies */
extern unsigned long marie_walker_interval_low;		/* jiffies */
extern unsigned long marie_walker_interval_normal;	/* jiffies */
extern unsigned long marie_walker_interval_idle;	/* jiffies */
extern unsigned int  marie_low_swappiness_mode;		/* bool; clamp effective swappiness to <=1 */

#ifdef CONFIG_SWAP
/*
 * kcompressd mode (sysfs /sys/kernel/mm/lru_marie/kcompressd):
 *   signed -100..+100, default +24.
 *
 *     0          — disabled (kthread fan-out off, swap_writeout inline)
 *     +1..+100   — Marie-gated. |value| is the queue depth at which the
 *                  producer treats the kfifo as full and falls back to
 *                  synchronous writeout.
 *     -1..-100   — force mode. Same queue-length semantics; runs even
 *                  when Marie is off.
 *
 * Default +24 mirrors the queue length kcompressd-unofficial proved
 * sound under sustained anon pressure. The producer reads vm_kcompressd
 * directly to derive the queue depth; the on/off and Marie/force gates
 * are encoded as a pair of static branches so the hot path in
 * mm/page_io.c::kcompressd_store costs a single jump in the common case:
 *
 *   kcompressd_enabled_key — true when vm_kcompressd != 0  (default TRUE)
 *   kcompressd_force_key   — true when vm_kcompressd  < 0  (default FALSE)
 *
 * The Marie-gated branch (positive value) reuses lru_marie_enabled_key
 * directly, so no extra branch is paid when Marie is on.
 */
DECLARE_STATIC_KEY_TRUE(kcompressd_enabled_key);
DECLARE_STATIC_KEY_FALSE(kcompressd_force_key);

extern int vm_kcompressd;

/**
 * kcompressd_active - should kswapd off-load this swap-out to kcompressd?
 *
 * Default-on: the enabled_key starts TRUE for the +24 default. Setting
 * vm_kcompressd to 0 flips it off; negative values force-on regardless
 * of Marie; positive values gate on lru_marie_enabled_key.
 */
static inline bool kcompressd_active(void)
{
	if (!static_branch_likely(&kcompressd_enabled_key))
		return false;
	if (static_branch_unlikely(&kcompressd_force_key))
		return true;
	return lru_marie_enabled();
}
#endif /* CONFIG_SWAP */

/*
 * Marie's per-folio state lives entirely in the per-PFN byte
 * marie_state[pfn] (declared in mm/lru_marie/state.h). Public callers
 * reach Marie state via the dispatch surface above
 * (lru_marie_add_folio / lru_marie_del_folio / lru_marie_shrink_lruvec /
 * lru_marie_look_around / lru_marie_age_node).
 * folio->flags carries no Marie bits.
 */

#endif /* CONFIG_LRU_MARIE */

/*
 * CONFIG_LRU_MARIE=n: this header intentionally exposes NO inline
 * shims. Every call site in mm/ is wrapped in #ifdef CONFIG_LRU_MARIE,
 * so when Marie is off the kernel image contains no Marie symbols and
 * no Marie calls at all. Refusing to provide no-ops here makes any
 * stray, un-gated reference fail to compile loudly rather than silently
 * disappearing into a return-false stub.
 */

#endif /* _LINUX_LRU_MARIE_H */
