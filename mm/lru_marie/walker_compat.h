/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_WALKER_COMPAT_H
#define _MM_LRU_MARIE_WALKER_COMPAT_H

/*
 * Per-kernel-version adaptation layer for the page-table walker (walker.c).
 *
 * Marie's core sources are meant to be byte-identical across every kernel it is
 * ported to (currently 6.12 / 6.18 / 7.0 / 7.1).  The only core code that
 * genuinely has to differ per version is a handful of in-tree mm APIs that
 * changed signature; the walker's share of those is isolated here behind
 * uniform names (the reclaim side lives in state_compat.h), so producing a
 * per-version patch re-touches just these small headers plus the unavoidable
 * context lines of the integration hunks -- never the bulk of the core.
 *
 * Include AFTER "../internal.h" (and the usual mm headers): the wrappers are
 * static inline and need the underlying declarations + struct types complete.
 *
 * Version boundaries below match the four supported targets exactly; revisit
 * them when adding a new target kernel.
 */

#include <linux/version.h>
#include <linux/mmu_notifier.h>	/* ptep_clear_young_notify (young-ptes shim) */

/*
 * pmd_devmap(): "this huge PMD maps ZONE_DEVICE memory". The walker's
 * PMD-mapped-THP path excludes those, mirroring MGLRU's get_pmd_pfn().
 *
 * Removed in 6.17 along with the rest of the DEVMAP page-table bit, so on
 * newer kernels the concept has no representation to test and the guard
 * collapses to a constant -- which is correct rather than merely
 * convenient: with no DEVMAP bit, no PMD can be one.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 17, 0)
#define marie_pmd_devmap(pmd)	pmd_devmap(pmd)
#else
#define marie_pmd_devmap(pmd)	0
#endif

/*
 * Look-around neighbour batching.  6.12 has folio_pte_batch() with the long
 * (max_nr, FPB flags, out-params) signature; 6.18+ replaced it with
 * folio_pte_batch_flags(folio, vma, ptep, &pte, max_nr, FPB_*).  Both collapse
 * a run of present PTEs mapping @folio while ignoring young/dirty differences:
 *   - 6.12: FPB_MERGE_YOUNG_DIRTY does not exist, but folio_pte_batch()
 *     pte_mkold()s before comparing, so FPB_IGNORE_DIRTY alone gives the same
 *     young/dirty-agnostic batching.
 *   - 6.18+: FPB_MERGE_YOUNG_DIRTY merges across young/dirty directly.
 * The uniform wrapper takes the 6.12 argument set (@addr is unused on 6.18+).
 */
static inline int
marie_folio_pte_batch(struct folio *folio, unsigned long addr, pte_t *ptep,
		      pte_t pte, unsigned int max_nr)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 18, 0)
	return folio_pte_batch_flags(folio, NULL, ptep, &pte, max_nr,
				     FPB_MERGE_YOUNG_DIRTY);
#else
	return folio_pte_batch(folio, addr, ptep, pte, max_nr, FPB_IGNORE_DIRTY,
			       NULL, NULL, NULL);
#endif
}

/*
 * arch_enter/leave_lazy_mmu_mode() were renamed to
 * lazy_mmu_mode_enable/disable() in 7.0.  Provide the new names on the older
 * targets so the walker can use one spelling.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 0, 0)
#define lazy_mmu_mode_enable()	arch_enter_lazy_mmu_mode()
#define lazy_mmu_mode_disable()	arch_leave_lazy_mmu_mode()
#endif

/*
 * test_and_clear_young_ptes_notify() -- the batched young-clear with
 * mmu-notifier callback -- landed in 7.1.  On older targets emulate it as a
 * per-PTE ptep_clear_young_notify() loop; functionally equivalent, losing only
 * the batched-notifier amortisation.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 1, 0)
static inline int marie_test_and_clear_young_ptes(struct vm_area_struct *vma,
						  unsigned long addr,
						  pte_t *pte, unsigned int nr)
{
	int young = 0;
	unsigned int i;

	for (i = 0; i < nr; i++)
		young |= ptep_clear_young_notify(vma, addr + i * PAGE_SIZE,
						 pte + i);
	return young;
}
#define test_and_clear_young_ptes_notify marie_test_and_clear_young_ptes
#endif

#endif /* _MM_LRU_MARIE_WALKER_COMPAT_H */
