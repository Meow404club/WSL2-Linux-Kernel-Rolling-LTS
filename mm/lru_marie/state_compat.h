/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_STATE_COMPAT_H
#define _MM_LRU_MARIE_STATE_COMPAT_H

/*
 * Per-kernel-version adaptation layer for the reclaim state machine (state.c).
 *
 * Marie's core sources are meant to be byte-identical across every kernel it is
 * ported to (currently 6.12 / 6.18 / 7.0 / 7.1).  The only core code that
 * genuinely has to differ per version is a handful of in-tree mm APIs that
 * changed signature; the reclaim side's share of those is isolated here behind
 * uniform names (the walker side lives in walker_compat.h), so producing a
 * per-version patch re-touches just these small headers plus the unavoidable
 * context lines of the integration hunks -- never the bulk of the core.
 *
 * Include AFTER "../internal.h" and the usual mm headers (in particular
 * <linux/vmstat.h> and <linux/memcontrol.h>): the wrappers are static inline
 * and need the underlying declarations + struct types complete.
 *
 * Version boundaries below match the four supported targets exactly; revisit
 * them when adding a new target kernel.
 */

#include <linux/version.h>

/*
 * folio->flags became the typed memdesc_flags_t -- struct { unsigned long f; }
 * -- in 6.18 (upstream series "Add and use memdesc_flags_t", first commit
 * 53fbef56e07d; <6.18 is a plain unsigned long, so this boundary is exact, not
 * just target-derived).  The macro yields the raw "unsigned long" lvalue on
 * every version, so both a read (`MARIE_FOLIO_FLAGS(f) & MASK`) and a bit op
 * (`set_mask_bits(&MARIE_FOLIO_FLAGS(f), ...)`) are version-agnostic.
 *
 * Marie touches the raw word in only the two narrow spots the page-flag
 * accessors do not cover: clearing stale LRU_GEN/LRU_REFS residue and the
 * single atomic PG_active->0 + PG_lru->1 publish.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 18, 0)
#define MARIE_FOLIO_FLAGS(folio)	((folio)->flags.f)
#else
#define MARIE_FOLIO_FLAGS(folio)	((folio)->flags)
#endif

/*
 * shrink_folio_list() gained a trailing @memcg parameter in 6.18 (the scan is
 * told which memcg it is reclaiming for).  Marie always has the memcg in hand
 * at the call site, so the uniform wrapper takes it unconditionally and simply
 * does not forward it on the pre-6.18 signature.
 */
static inline unsigned int
marie_shrink_folio_list(struct list_head *folio_list, struct pglist_data *pgdat,
			struct scan_control *sc, struct reclaim_stat *stat,
			bool ignore_references, struct mem_cgroup *memcg)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 18, 0)
	return shrink_folio_list(folio_list, pgdat, sc, stat, ignore_references,
				 memcg);
#else
	return shrink_folio_list(folio_list, pgdat, sc, stat, ignore_references);
#endif
}

/*
 * Reclaim-counter accounting.
 *
 * 7.1 relocated the PGSTEAL, PGSCAN, PGDEMOTE and PGREFILL counters out of
 * enum vm_event_item into enum node_stat_item (they are per-memcg lruvec stats
 * now).  Route the post-isolation (PGSCAN) and post-reclaim (PGSTEAL) bumps
 * through whichever API the building kernel exposes.  @base is the
 * PG{SCAN,STEAL}_KSWAPD reclaimer-offset base, @per_type the PG{SCAN,STEAL}_ANON
 * counter indexed by the anon/file @type; both are passed as int so the call
 * site is identical whichever enum they live in.
 */
static inline void marie_account_reclaim(struct lruvec *lruvec,
					 struct scan_control *sc,
					 int base, int per_type,
					 int type, unsigned long nr)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 1, 0)
	/* node_stat_item: one lruvec update folds node vmstat + memcg stat. */
	mod_lruvec_state(lruvec, base + vmscan_reclaimer_offset(sc), nr);
	mod_lruvec_state(lruvec, per_type + type, nr);
#else
	/* vm_event_item: global (skipped for cgroup reclaim) + memcg + type. */
	int item = base + vmscan_reclaimer_offset(sc);

	if (!sc_cgroup_reclaim(sc))
		count_vm_events(item, nr);
	count_memcg_events(lruvec_memcg(lruvec), item, nr);
	count_vm_events(per_type + type, nr);
#endif
}

#endif /* _MM_LRU_MARIE_STATE_COMPAT_H */
