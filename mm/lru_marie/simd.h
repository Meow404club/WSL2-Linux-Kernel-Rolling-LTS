/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _MM_LRU_MARIE_SIMD_H
#define _MM_LRU_MARIE_SIMD_H

/*
 * Marie SIMD-accelerated PTE scan.
 *
 * Single entry point:
 *
 *      lru_marie_simd_young_pte_mask(pte_table, bitmap);
 *
 * On x86 it wraps one PMD's scan in kernel_fpu_begin/end; per-PMD
 * begin/end is effectively free (kthreads skip the FPU save, the restore
 * is deferred to userspace return), so there is nothing to amortise
 * across PMDs. On every other arch (including arm64) the scan is a plain
 * scalar pte_young loop with no FPU state.
 */

#include <linux/types.h>
#include <linux/bits.h>
#include <linux/jump_label.h>

#ifdef CONFIG_X86
/*
 * Runtime kill-switch for the boot-detected SIMD walker, exposed via
 * /sys/kernel/mm/lru_marie/simd. Default true: walker uses the widest
 * SIMD kernel that arch_initcall could pick (AVX-512F > AVX2 > SSE2).
 * Writing 0 to the sysfs file flips the static branch so the walker
 * falls back to a pure scalar pte_young loop in the same translation
 * unit.
 *
 * Other arches use the generic scalar fallback already, so the toggle
 * does not need to exist there and the sysfs attribute is hidden.
 */
DECLARE_STATIC_KEY_TRUE(marie_simd_enabled_key);

static inline bool marie_simd_enabled(void)
{
	return static_branch_likely(&marie_simd_enabled_key);
}

/*
 * SIMD ISA cap (simd_max) accessors. Implemented in simd_x86.c, consumed by
 * the /sys/kernel/mm/lru_marie/simd_max knob in core.c. marie_simd_max_name()
 * returns the current cap ("avx512"/"avx2"/"sse2"); marie_simd_max_store()
 * parses a name, applies it (re-patching the scan static call), and returns 0
 * or -EINVAL.
 */
const char *marie_simd_max_name(void);
int marie_simd_max_store(const char *buf);
#else
static inline bool marie_simd_enabled(void) { return false; }
#endif

/*
 * Number of unsigned longs needed to hold the young-bit bitmap for one
 * PMD's worth of PTEs (PTRS_PER_PTE = 512 on x86_64; the value is
 * pulled from the arch's pgtable headers via the caller's includes).
 */
#define MARIE_SIMD_PTE_BITMAP_LONGS	((512 + BITS_PER_LONG - 1) / BITS_PER_LONG)

/**
 * lru_marie_simd_young_pte_mask - scan one PMD's PTE array for young bits.
 * @table:  pointer to the first pte_t in the PMD's PTE array (512 entries)
 * @bitmap: output, MARIE_SIMD_PTE_BITMAP_LONGS unsigned longs.
 *
 * The sole entry point. On x86 it self-brackets the scan in
 * kernel_fpu_begin/end (or runs the scalar loop when the simd knob is
 * off); on other arches it is a plain scalar pte_young loop.
 */
void lru_marie_simd_young_pte_mask(const void *table, unsigned long *bitmap);

#endif /* _MM_LRU_MARIE_SIMD_H */
