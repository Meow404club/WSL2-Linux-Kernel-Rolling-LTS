/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_DEFRAG_COMPAT_H
#define _MM_LRU_MARIE_DEFRAG_COMPAT_H

/*
 * Per-kernel-version adaptation layer for the defragmenter (defrag.c).
 *
 * Marie's core sources are meant to be byte-identical across every kernel it is
 * ported to (currently 6.12 / 6.18 / 7.0 / 7.1 / 7.2).  The only defrag code
 * that genuinely has to differ per version is isolated here behind a uniform
 * name so a per-version patch never has to touch defrag.c itself: the free-page
 * allocation contract that changed in 6.14, and the has_unmovable_pages()
 * signature that changed in 6.17.
 *
 * Include AFTER "../internal.h" (and the usual mm headers): the wrappers are
 * static inline and need post_alloc_hook()/set_page_refcounted() +
 * __GFP_MOVABLE declared.
 *
 * Version boundary below matches the supported targets; revisit when adding a
 * new target kernel.
 */

#include <linux/version.h>
#include <linux/page-isolation.h>	/* enum pb_isolate_mode (>= 6.17) */

/*
 * marie_defrag_prep_allocated - turn a harvested raw free page (refcount 0,
 * removed from the buddy allocator via __isolate_free_page()) into a
 * fully-allocated, refcounted page.  This is the free-pool analogue of stock
 * compaction's mark_allocated() and the post_alloc_hook() sequence in
 * compaction_alloc(): both the migration-target handout (alloc_target) and the
 * leftover-return-to-buddy path (freectx_release) need it.
 *
 * Up to 6.13 post_alloc_hook() itself set the refcount to 1.  6.14 moved that
 * to the callers (upstream: set_page_refcounted() split out of
 * post_alloc_hook(); compaction_alloc()/mark_allocated() gained an explicit
 * call).  On >= 6.14 we must therefore add set_page_refcounted(); on <= 6.13
 * doing so would double-set it and trip VM_BUG_ON_PAGE(page_ref_count) under
 * CONFIG_DEBUG_VM -- hence the gate.
 *
 * A refcount-0 page escaping this helper is not benign: as a migration target
 * it is freed mid-migration by folio_add_lru() (bad_page on a still-locked /
 * just-remapped folio, refcount underflow, live-anon corruption); on the
 * release path __free_pages()' put_page_testzero() never fires, leaking the
 * pool back-pressure.
 */
static inline void marie_defrag_prep_allocated(struct page *page, unsigned int order)
{
	post_alloc_hook(page, order, __GFP_MOVABLE);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 14, 0)
	set_page_refcounted(page);
#endif
}

/**
 * marie_defrag_block_has_unmovable - can anything in [@start_pfn, @end_pfn) never
 * be moved out of the way?
 *
 * Wraps has_unmovable_pages(), whose signature changed in 6.17: the
 * (migratetype, isol_flags) pair became a single enum pb_isolate_mode. The
 * parameter is read in exactly three places on both sides of that change, and
 * they line up one to one:
 *
 *	<= 6.16                          >= 6.17
 *	is_migrate_cma(migratetype)      mode == PB_ISOLATE_MODE_CMA_ALLOC
 *	isol_flags & MEMORY_OFFLINE      mode == PB_ISOLATE_MODE_MEM_OFFLINE   (x2)
 *
 * Marie is neither offlining memory nor allocating CMA -- it is asking whether a
 * candidate source block could ever be emptied -- so it passed
 * (MIGRATE_MOVABLE, 0), which makes both tests false. PB_ISOLATE_MODE_OTHER is
 * the value that makes both tests false on the new side, so the two calls are
 * exactly equivalent rather than merely close.
 *
 * Returns true if the range holds something unmovable. The underlying function
 * returns the offending page; the caller only ever asks the yes/no question.
 */
static inline bool marie_defrag_block_has_unmovable(unsigned long start_pfn,
						    unsigned long end_pfn)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 17, 0)
	return has_unmovable_pages(start_pfn, end_pfn, PB_ISOLATE_MODE_OTHER);
#else
	return has_unmovable_pages(start_pfn, end_pfn, MIGRATE_MOVABLE, 0);
#endif
}

/*
 * PORTING defrag.c TO A NEW KERNEL
 * ===============================
 *
 * The rule: defrag.c stays BYTE-IDENTICAL on every branch. Never add a
 * LINUX_VERSION_CODE test to it. Every version delta belongs in this header,
 * behind a name defrag.c already calls. Porting is then "copy defrag.c and
 * defrag_compat.h across verbatim", and a diff between two branches that shows
 * anything in defrag.c is a bug in the port, not a necessary adaptation.
 *
 * What to re-check when adding a target kernel, in order of how badly it bites:
 *
 *  1. The free-page allocation contract -- marie_defrag_prep_allocated() above.
 *     This is the one that corrupts memory silently rather than failing to
 *     build. Confirm which side of the split the target is on by reading
 *     post_alloc_hook() and stock compaction_alloc()/mark_allocated(): if
 *     compaction_alloc() calls set_page_refcounted() explicitly, so must we.
 *     Getting it wrong on >= 6.14 hands refcount-0 pages to migrate_pages(),
 *     and the symptom is a storm of "Bad page state" during kcompactd with
 *     stacks through migrate_folio_move -> folio_add_lru -> free_unref_folios,
 *     plus live anon corruption. It was found the hard way on 7.1.
 *
 *  1b. The has_unmovable_pages() signature --
 *     marie_defrag_block_has_unmovable() above. Changed in 6.17 and, unlike the
 *     refcount contract, this one fails to BUILD rather than silently corrupting,
 *     so it is the pleasant kind. The declaration added to mm/internal.h by the
 *     page_isolation patch has to match the target's signature too; that hunk is
 *     outside this header and is reviewed per version anyway.
 *
 *  2. has_unmovable_pages() -- defrag.c calls it (declared in mm/internal.h) to
 *     reject source blocks that evacuation could never free, and Marie is a
 *     LOCKLESS caller where page_isolation's own caller holds zone->lock. Two
 *     things to confirm on a new target: that it is still non-static (the
 *     branch carries that change), and that its internal buddy_order() read is
 *     still the lockless-safe buddy_order_unsafe() form with the order clamped
 *     before shifting. An upstream rebase that reverts either will build fine
 *     and then misbehave under a racing buddy split. CONFIG_MEMORY_ISOLATION
 *     must also still be selected by LRU_MARIE_DEFRAG in mm/Kconfig, or
 *     page_isolation.o is not built and the link fails.
 *
 *  3. The kcompactd request fields marie_defrag_topn() reads to decide when the
 *     demand path can stop early: pgdat->kcompactd_max_order,
 *     pgdat->kcompactd_highest_zoneidx, and zone_watermark_ok() /
 *     low_wmark_pages(). These have been stable for a long time; if one is
 *     renamed the build breaks loudly, which is the safe kind of failure.
 *
 *  4. The two mm/compaction.c hook sites that hand kcompactd's work to Marie
 *     (kcompactd_do_work's early return, and the compact_node() swap on the
 *     proactive path). Those live in the vmscan/compaction patch hunk, not
 *     here, and are the one place a port genuinely has to re-apply by hand.
 */

#endif /* _MM_LRU_MARIE_DEFRAG_COMPAT_H */
