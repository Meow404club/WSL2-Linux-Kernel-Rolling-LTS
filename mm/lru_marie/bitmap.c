// SPDX-License-Identifier: GPL-2.0
/*
 * Hierarchical PFN bitmap operations. See bitmap.h for the design
 * overview. Used by the global per-(type, zone, gen) plane.
 */

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/bitops.h>
#include <linux/cache.h>
#include <linux/export.h>
#include <linux/kernel.h>
#include <linux/log2.h>
#include <linux/minmax.h>
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/slab.h>

#include "bitmap.h"

unsigned int marie_max_l2_pages_per_bit = MARIE_MAX_L2_PAGES_PER_BIT_DEFAULT;
unsigned int marie_l2_shift;
unsigned int marie_l2_nbits;

/*
 * Dynamically sized (marie_l2_nbits entries), unlike a fixed compile-time
 * array: marie_l2_nbits is a boot-time value (from marie_max_l2_pages_per_bit),
 * not a compile-time constant.
 */
struct marie_bm_range_lock *marie_bm_range_locks;

/*
 * marie_bm_global_init - derive marie_l2_shift/marie_l2_nbits from max_pfn
 * and the configured marie_max_l2_pages_per_bit ceiling, then allocate the
 * shared range-lock array. Must run before any marie_bm_init() call.
 *
 * Picks the finer of two candidate shifts (see MARIE_L2_DEFAULT_PAGES_
 * PER_BIT in bitmap.h for the rationale):
 *   - shift_native: max_pfn spread evenly over MARIE_L2_TARGET_BITS (512).
 *     Finer than the ceiling whenever the system is small enough that
 *     512 bits alone already give a per-bit page count at or under the
 *     ceiling -- native wins, and the plane stays at (approximately)
 *     512 bits.
 *   - shift_ceiling: the configured (or default) marie_max_l2_pages_per_bit,
 *     rounded up to a power of two. Wins once max_pfn is large enough
 *     that native granularity would exceed the ceiling -- the plane then
 *     grows past 512 bits instead of letting pages/bit exceed the ceiling.
 * Both candidates round UP to the next power of two (order_base_2) for a
 * plain shift in the hot path, so the actual pages/bit for whichever
 * candidate wins can end up somewhat larger than its unrounded input.
 */
int marie_bm_global_init(void)
{
	unsigned long ppb_ceiling = marie_max_l2_pages_per_bit;
	unsigned long ppb_native;
	unsigned int shift_native, shift_ceiling;

	if (!max_pfn)
		return 0;
	if (!ppb_ceiling)
		ppb_ceiling = MARIE_MAX_L2_PAGES_PER_BIT_DEFAULT;

	ppb_native = DIV_ROUND_UP(max_pfn, MARIE_L2_TARGET_BITS);
	if (!ppb_native)
		ppb_native = 1;

	shift_native = order_base_2(ppb_native);
	shift_ceiling = order_base_2(ppb_ceiling);
	marie_l2_shift = min(shift_native, shift_ceiling);

	marie_l2_nbits = DIV_ROUND_UP(max_pfn, 1UL << marie_l2_shift);
	if (!marie_l2_nbits)
		marie_l2_nbits = 1;

	/*
	 * __GFP_ZERO is the entire init: atomic_t's zero value is already
	 * "unclaimed" (marie_bm_range_trylock's cmpxchg(&claimed, 0, 1)), so
	 * no per-entry init loop is needed the way spin_lock_init() used to
	 * require.
	 */
	marie_bm_range_locks = kvmalloc_array(marie_l2_nbits,
					      sizeof(*marie_bm_range_locks),
					      GFP_KERNEL | __GFP_ZERO);
	if (!marie_bm_range_locks)
		return -ENOMEM;

	pr_info("L2: %u bits (%lu pages/bit, max_pfn=%lu)\n",
		marie_l2_nbits, 1UL << marie_l2_shift, max_pfn);
	return 0;
}

int marie_bm_init(struct marie_bitmap *bm)
{
	unsigned long bytes;

	if (!max_pfn)
		return 0;

	bytes = BITS_TO_LONGS(max_pfn) * sizeof(unsigned long);
	bm->l1 = kvmalloc(bytes, GFP_KERNEL | __GFP_ZERO);
	if (!bm->l1)
		return -ENOMEM;

	bytes = BITS_TO_LONGS(marie_l2_nbits) * sizeof(unsigned long);
	bm->l2 = kvmalloc(bytes, GFP_KERNEL | __GFP_ZERO);
	if (!bm->l2) {
		kvfree(bm->l1);
		bm->l1 = NULL;
		return -ENOMEM;
	}

	bm->l2_count = kvmalloc_array(marie_l2_nbits, sizeof(*bm->l2_count),
				     GFP_KERNEL | __GFP_ZERO);
	if (!bm->l2_count) {
		kvfree(bm->l2);
		bm->l2 = NULL;
		kvfree(bm->l1);
		bm->l1 = NULL;
		return -ENOMEM;
	}

	return 0;
}

void marie_bm_free(struct marie_bitmap *bm)
{
	if (!bm)
		return;
	kvfree(bm->l1);
	bm->l1 = NULL;
	kvfree(bm->l2);
	bm->l2 = NULL;
	kvfree(bm->l2_count);
	bm->l2_count = NULL;
}

/*
 * marie_bm_set / marie_bm_clear / marie_bm_test are static inline in
 * bitmap.h -- they sit on the install / del / promote hot path and
 * out-of-lining costs measurable cycles per fault.
 */

/*
 * Inclusive [start_word, end_word) covering one L2 bit's worth of L1 words.
 * Clipped to the actual l1 storage extent.
 */
static void marie_bm_l1_word_range(unsigned int l2bit,
				   unsigned long *start_word,
				   unsigned long *end_word)
{
	unsigned long pfns_per_l2 = 1UL << marie_l2_shift;
	unsigned long start_pfn = (unsigned long)l2bit << marie_l2_shift;
	unsigned long end_pfn = start_pfn + pfns_per_l2;
	unsigned long max_words = BITS_TO_LONGS(max_pfn);

	*start_word = start_pfn / BITS_PER_LONG;
	*end_word = DIV_ROUND_UP(end_pfn, BITS_PER_LONG);
	if (*end_word > max_words)
		*end_word = max_words;
}

void marie_bm_drop_l2_range(struct marie_bitmap *bm, unsigned int l2bit)
{
	unsigned long start_word, end_word, wi;

	if (!bm->l1)
		return;
	marie_bm_l1_word_range(l2bit, &start_word, &end_word);
	for (wi = start_word; wi < end_word; wi++)
		bm->l1[wi] = 0;
	atomic_set(&bm->l2_count[l2bit], 0);
	clear_bit(l2bit, bm->l2);
}

void marie_bm_reset(struct marie_bitmap *bm)
{
	unsigned int i;

	if (!bm->l1)
		return;
	if (max_pfn)
		bitmap_zero(bm->l1, max_pfn);
	bitmap_zero(bm->l2, marie_l2_nbits);
	for (i = 0; i < marie_l2_nbits; i++)
		atomic_set(&bm->l2_count[i], 0);
}
