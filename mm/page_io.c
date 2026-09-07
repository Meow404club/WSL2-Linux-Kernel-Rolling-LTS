// SPDX-License-Identifier: GPL-2.0
/*
 *  linux/mm/page_io.c
 *
 *  Copyright (C) 1991, 1992, 1993, 1994  Linus Torvalds
 *
 *  Swap reorganised 29.12.95, 
 *  Asynchronous swapping added 30.12.95. Stephen Tweedie
 *  Removed race in async swapping. 14.4.1996. Bruno Haible
 *  Add swap of shared pages through the page cache. 20.2.1998. Stephen Tweedie
 *  Always use brw_page, life becomes simpler. 12 May 1998 Eric Biederman
 */

#include <linux/mm.h>
#include <linux/kernel_stat.h>
#include <linux/gfp.h>
#include <linux/pagemap.h>
#include <linux/swap.h>
#include <linux/bio.h>
#include <linux/swapops.h>
#include <linux/writeback.h>
#include <linux/blkdev.h>
#include <linux/psi.h>
#include <linux/uio.h>
#include <linux/sched/task.h>
#include <linux/delayacct.h>
#include <linux/zswap.h>
#include <linux/kfifo.h>
#include <linux/lru_marie.h>
#include "swap.h"

#ifdef CONFIG_LRU_MARIE
/*
 * Counter consumed by the early-OOM gate in
 * mm/page_alloc.c:should_reclaim_retry. Declared in include/linux/swap.h.
 * Marie-only: omitted entirely under CONFIG_LRU_MARIE=n.
 */
atomic_long_t nr_swap_write_failed = ATOMIC_LONG_INIT(0);
#endif

static void __end_swap_bio_write(struct bio *bio)
{
	struct folio *folio = bio_first_folio_all(bio);

	if (bio->bi_status) {
		/*
		 * We failed to write the page out to swap-space.
		 * Re-dirty the page in order to avoid it being reclaimed.
		 * Also print a dire warning that things will go BAD (tm)
		 * very quickly.
		 *
		 * Also clear PG_reclaim to avoid folio_rotate_reclaimable()
		 *
		 * Bump nr_swap_write_failed so the early-OOM gate in
		 * should_reclaim_retry can short-circuit the
		 * MAX_RECLAIM_RETRIES wait when the swap backend (most
		 * commonly ZRAM/zswap zs_malloc, or a real disk error) has
		 * stopped accepting writes — anon reclaim is doomed in that
		 * state regardless of get_nr_swap_pages() reporting free
		 * entries. Marie-only signal; vanilla MGLRU/Legacy builds
		 * (lru_marie_enabled()=false) skip the counter bump so the
		 * baseline allocator sees vanilla retry behaviour.
		 */
#ifdef CONFIG_LRU_MARIE
		if (lru_marie_enabled())
			atomic_long_inc(&nr_swap_write_failed);
#endif
		folio_mark_dirty(folio);
		pr_alert_ratelimited("Write-error on swap-device (%u:%u:%llu)\n",
				     MAJOR(bio_dev(bio)), MINOR(bio_dev(bio)),
				     (unsigned long long)bio->bi_iter.bi_sector);
		folio_clear_reclaim(folio);
	}
	folio_end_writeback(folio);
}

#ifdef CONFIG_LRU_MARIE
/* Multi-folio variant of __end_swap_bio_write() for coalesced swap bios. */
static void __end_swap_bio_write_batch(struct bio *bio)
{
	struct folio_iter fi;

	if (bio->bi_status) {
		bio_for_each_folio_all(fi, bio) {
			if (lru_marie_enabled())
				atomic_long_inc(&nr_swap_write_failed);
			folio_mark_dirty(fi.folio);
			folio_clear_reclaim(fi.folio);
		}
		pr_alert_ratelimited("Write-error on swap-device (%u:%u:%llu)\n",
			MAJOR(bio_dev(bio)), MINOR(bio_dev(bio)),
			(unsigned long long)bio->bi_iter.bi_sector);
	}
	bio_for_each_folio_all(fi, bio)
		folio_end_writeback(fi.folio);
}
#endif /* CONFIG_LRU_MARIE */

static void end_swap_bio_write(struct bio *bio)
{
	__end_swap_bio_write(bio);
	bio_put(bio);
}

static void __end_swap_bio_read(struct bio *bio)
{
	struct folio *folio = bio_first_folio_all(bio);

	if (bio->bi_status) {
		pr_alert_ratelimited("Read-error on swap-device (%u:%u:%llu)\n",
				     MAJOR(bio_dev(bio)), MINOR(bio_dev(bio)),
				     (unsigned long long)bio->bi_iter.bi_sector);
	} else {
		folio_mark_uptodate(folio);
	}
	folio_unlock(folio);
}

static void end_swap_bio_read(struct bio *bio)
{
	__end_swap_bio_read(bio);
	bio_put(bio);
}

int generic_swapfile_activate(struct swap_info_struct *sis,
				struct file *swap_file,
				sector_t *span)
{
	struct address_space *mapping = swap_file->f_mapping;
	struct inode *inode = mapping->host;
	unsigned blocks_per_page;
	unsigned long page_no;
	unsigned blkbits;
	sector_t probe_block;
	sector_t last_block;
	sector_t lowest_block = -1;
	sector_t highest_block = 0;
	int nr_extents = 0;
	int ret;

	blkbits = inode->i_blkbits;
	blocks_per_page = PAGE_SIZE >> blkbits;

	/*
	 * Map all the blocks into the extent tree.  This code doesn't try
	 * to be very smart.
	 */
	probe_block = 0;
	page_no = 0;
	last_block = i_size_read(inode) >> blkbits;
	while ((probe_block + blocks_per_page) <= last_block &&
			page_no < sis->max) {
		unsigned block_in_page;
		sector_t first_block;

		cond_resched();

		first_block = probe_block;
		ret = bmap(inode, &first_block);
		if (ret || !first_block)
			goto bad_bmap;

		/*
		 * It must be PAGE_SIZE aligned on-disk
		 */
		if (first_block & (blocks_per_page - 1)) {
			probe_block++;
			goto reprobe;
		}

		for (block_in_page = 1; block_in_page < blocks_per_page;
					block_in_page++) {
			sector_t block;

			block = probe_block + block_in_page;
			ret = bmap(inode, &block);
			if (ret || !block)
				goto bad_bmap;

			if (block != first_block + block_in_page) {
				/* Discontiguity */
				probe_block++;
				goto reprobe;
			}
		}

		first_block >>= (PAGE_SHIFT - blkbits);
		if (page_no) {	/* exclude the header page */
			if (first_block < lowest_block)
				lowest_block = first_block;
			if (first_block > highest_block)
				highest_block = first_block;
		}

		/*
		 * We found a PAGE_SIZE-length, PAGE_SIZE-aligned run of blocks
		 */
		ret = add_swap_extent(sis, page_no, 1, first_block);
		if (ret < 0)
			goto out;
		nr_extents += ret;
		page_no++;
		probe_block += blocks_per_page;
reprobe:
		continue;
	}
	ret = nr_extents;
	*span = 1 + highest_block - lowest_block;
	if (page_no == 0)
		page_no = 1;	/* force Empty message */
	sis->max = page_no;
	sis->pages = page_no - 1;
out:
	return ret;
bad_bmap:
	pr_err("swapon: swapfile has holes\n");
	ret = -EINVAL;
	goto out;
}

static bool is_folio_zero_filled(struct folio *folio)
{
	unsigned int pos, last_pos;
	unsigned long *data;
	unsigned int i;

	last_pos = PAGE_SIZE / sizeof(*data) - 1;
	for (i = 0; i < folio_nr_pages(folio); i++) {
		data = kmap_local_folio(folio, i * PAGE_SIZE);
		/*
		 * Check last word first, incase the page is zero-filled at
		 * the start and has non-zero data at the end, which is common
		 * in real-world workloads.
		 */
		if (data[last_pos]) {
			kunmap_local(data);
			return false;
		}
		for (pos = 0; pos < last_pos; pos++) {
			if (data[pos]) {
				kunmap_local(data);
				return false;
			}
		}
		kunmap_local(data);
	}

	return true;
}

static void swap_zeromap_folio_set(struct folio *folio)
{
	struct obj_cgroup *objcg = get_obj_cgroup_from_folio(folio);
	struct swap_info_struct *sis = __swap_entry_to_info(folio->swap);
	int nr_pages = folio_nr_pages(folio);
	swp_entry_t entry;
	unsigned int i;

	for (i = 0; i < folio_nr_pages(folio); i++) {
		entry = page_swap_entry(folio_page(folio, i));
		set_bit(swp_offset(entry), sis->zeromap);
	}

	count_vm_events(SWPOUT_ZERO, nr_pages);
	if (objcg) {
		count_objcg_events(objcg, SWPOUT_ZERO, nr_pages);
		obj_cgroup_put(objcg);
	}
}

static void swap_zeromap_folio_clear(struct folio *folio)
{
	struct swap_info_struct *sis = __swap_entry_to_info(folio->swap);
	swp_entry_t entry;
	unsigned int i;

	for (i = 0; i < folio_nr_pages(folio); i++) {
		entry = page_swap_entry(folio_page(folio, i));
		clear_bit(swp_offset(entry), sis->zeromap);
	}
}

/*
 * do_swapout() - Write a folio to swap space
 * @folio: The folio to write out
 *
 * This function writes the folio to swap space, either using zswap or
 * synchronous write. It ensures that the folio is unlocked and the
 * reference count is decremented after the operation.
 */
static inline void do_swapout(struct folio *folio, struct swap_iocb **swap_plug)
{
	if (zswap_store(folio)) {
		count_mthp_stat(folio_order(folio), MTHP_STAT_ZSWPOUT);
		folio_unlock(folio);
	} else
		__swap_writepage(folio, swap_plug); /* Implies folio_unlock(folio) */

	/* Decrement the folio reference count */
	folio_put(folio);
}

#ifdef CONFIG_LRU_MARIE
/* Forward decl: defined below, after swap_writepage_bdev_sync(). */
static void swap_writepage_bdev_sync_batch(struct folio **folios,
		unsigned int n, struct swap_info_struct *sis);

/*
 * do_swapout_batch() - Write a drained batch of folios to swap.
 *
 * Partitions the batch into runs eligible for bio-coalescing (same
 * swap_info_struct, all anon, zswap off, SWP_SYNCHRONOUS_IO) and
 * dispatches each such run through swap_writepage_bdev_sync_batch() --
 * one multi-segment bio per contiguous-slot run instead of one bio per
 * folio. Folios that don't qualify (zswap on, non-anon, non-sync-IO
 * device) fall back to the per-folio do_swapout() path unchanged.
 */
static void do_swapout_batch(struct folio **folios, unsigned int n,
			     struct swap_iocb **swap_plug)
{
	unsigned int i = 0;

	while (i < n) {
		struct folio *f = folios[i];
		struct swap_info_struct *sis = __swap_entry_to_info(f->swap);

		/* zswap on, non-anon, or non-sync-IO block dev -> per-folio. */
		if (zswap_is_enabled() || !folio_test_anon(f) ||
		    !data_race(sis->flags & SWP_SYNCHRONOUS_IO)) {
			do_swapout(f, swap_plug);
			i++;
			continue;
		}

		/* Build a same-sis run. */
		{
			unsigned int start = i;

			while (i < n) {
				struct folio *g = folios[i];

				if (__swap_entry_to_info(g->swap) != sis ||
				    !folio_test_anon(g) ||
				    zswap_is_enabled())
					break;
				i++;
			}
			swap_writepage_bdev_sync_batch(&folios[start], i - start,
						       sis);
			/* drop the refs kcompressd_store took (do_swapout would). */
			for (unsigned int j = start; j < i; j++)
				folio_put(folios[j]);
		}
	}
}

/*
 * kcompressd_store() - Off-load folio compression to kcompressd
 * @folio: The folio to compress
 *
 * This function attempts to off-load the compression of the folio to
 * kcompressd. If kcompressd is not available or the folio cannot be
 * compressed, it falls back to synchronous write.
 *
 * Returns true if the folio was successfully queued for compression,
 * false otherwise.
 */
static bool kcompressd_store(struct folio *folio, struct swap_iocb **swap_plug)
{
	pg_data_t *pgdat = NODE_DATA(numa_node_id());
	unsigned int ret;
	struct folio *head = NULL;

	/* Only kswapd can use kcompressd */
	if (!current_is_kswapd())
		return false;

	/* Mode 0, or mode 1 with Marie off — short-circuit on the static branches. */
	if (!kcompressd_active())
		return false;

	/* kthread must be running */
	if (unlikely(!pgdat->kcompressd))
		return false;

	/* We can only off-load anon folios */
	if (!folio_test_anon(folio))
		return false;

	/* Fall back to synchronously return AOP_WRITEPAGE_ACTIVATE.
	 * folio_memcg -> obj_cgroup_memcg requires RCU read-side held to
	 * keep objcg from being freed by a concurrent memcg teardown
	 * (lockdep_assert_once in obj_cgroup_memcg). */
	{
		bool zswap_wb_ok;

		rcu_read_lock();
		zswap_wb_ok = mem_cgroup_zswap_writeback_enabled(folio_memcg(folio));
		rcu_read_unlock();
		if (!zswap_wb_ok)
			return false;
	}

	/* Swap device must be sync-efficient */
	if (!zswap_is_enabled() &&
		!data_race(__swap_entry_to_info(folio->swap)->flags & SWP_SYNCHRONOUS_IO))
		return false;

	/*
	 * The kfifo backing storage is sized at KCOMPRESSD_FIFO_SIZE (the
	 * compile-time max). The effective queue depth is |vm_kcompressd|;
	 * when current depth meets or exceeds that, treat the queue as
	 * full and swap out the head folio synchronously to make space.
	 */
	scoped_guard(spinlock_irqsave, &pgdat->kcompressd_fifo_lock)
		if (kfifo_len(&pgdat->kcompressd_fifo) >=
			abs(READ_ONCE(vm_kcompressd)) * sizeof(struct folio *) &&
			unlikely(!kfifo_out(&pgdat->kcompressd_fifo,
					&head, sizeof(folio))))
			return false;

	/* Increment the folio reference count to avoid it being freed */
	folio_get(folio);

	/* Enqueue the folio for compression */
	ret = kfifo_in(&pgdat->kcompressd_fifo, &folio, sizeof(folio));
	if (likely(ret))
		/* We successfully enqueued the folio. wake up kcompressd */
		wake_up_interruptible(&pgdat->kcompressd_wait);
	else
		/* Enqueue failed, so we must cancel the reference count */
		folio_put(folio);

	/* If we had to swap out the head folio, do it now.
	 * This will block until the folio is written out.
	 */
	if (head)
		do_swapout(head, swap_plug);

	return ret;
}
#else  /* !CONFIG_LRU_MARIE */
static inline bool kcompressd_store(struct folio *folio,
				   struct swap_iocb **swap_plug)
{
	return false;
}
#endif

/*
 * We may have stale swap cache pages in memory: notice
 * them here and get rid of the unnecessary final write.
 */
int swap_writeout(struct folio *folio, struct swap_iocb **swap_plug)
{
	int ret = 0;

	if (folio_free_swap(folio))
		goto out_unlock;

	/*
	 * Arch code may have to preserve more data than just the page
	 * contents, e.g. memory tags.
	 */
	ret = arch_prepare_to_swap(folio);
	if (ret) {
		folio_mark_dirty(folio);
		goto out_unlock;
	}

	/*
	 * Use a bitmap (zeromap) to avoid doing IO for zero-filled pages.
	 * The bits in zeromap are protected by the locked swapcache folio
	 * and atomic updates are used to protect against read-modify-write
	 * corruption due to other zero swap entries seeing concurrent updates.
	 */
	if (is_folio_zero_filled(folio)) {
		swap_zeromap_folio_set(folio);
		goto out_unlock;
	}

	/*
	 * Clear bits this folio occupies in the zeromap to prevent zero data
	 * being read in from any previous zero writes that occupied the same
	 * swap entries.
	 */
	swap_zeromap_folio_clear(folio);

	/*
	 * Compression within zswap and zram might block rmap, unmap
	 * of both file and anon pages, try to do compression async
	 * if possible
	 */
	if (kcompressd_store(folio, swap_plug))
		return 0;

	if (zswap_store(folio)) {
		count_mthp_stat(folio_order(folio), MTHP_STAT_ZSWPOUT);
		goto out_unlock;
	}
	if (!mem_cgroup_zswap_writeback_enabled(folio_memcg(folio))) {
		folio_mark_dirty(folio);
		return AOP_WRITEPAGE_ACTIVATE;
	}

	__swap_writepage(folio, swap_plug);
	return 0;
out_unlock:
	folio_unlock(folio);
	return ret;
}

#ifdef CONFIG_LRU_MARIE
/*
 * Batch-drain size for kcompressd(): reuses the existing |vm_kcompressd|
 * queue-depth knob (no separate batch-size knob) clamped to a sane range
 * for the stack-allocated drain array.
 */
static inline unsigned int kcompressd_drain_depth(void)
{
	int d = abs(READ_ONCE(vm_kcompressd));

	if (d < 1)
		return 1;
	if (d > SWAP_CLUSTER_MAX)
		return SWAP_CLUSTER_MAX;
	return d;
}

/*
 * kcompressd() - Kernel thread for compressing folios
 * @p: Pointer to pg_data_t structure
 *
 * This function runs in a kernel thread and waits for folios to be
 * queued for compression. It drains its fifo in batches and dispatches
 * each batch through do_swapout_batch(), which coalesces same-device
 * contiguous-slot folios into multi-segment bios (see Stage 1 above).
 */
int kcompressd(void *p)
{
	pg_data_t *pgdat = (pg_data_t *)p;
	struct folio *batch[SWAP_CLUSTER_MAX];
	unsigned int n;

	/* * kcompressd runs with PF_MEMALLOC and PF_KSWAPD flags set to
	 * allow it to allocate memory for compression without being
	 * restricted by the current memory allocation context.
	 * Also PF_KSWAPD prevents Intel Graphics driver from crashing
	 * the system in i915_gem_shrinker.c:i915_gem_shrinker_scan()
	 */
	current->flags |= PF_MEMALLOC | PF_KSWAPD;

	while (!kthread_should_stop()) {
		wait_event_interruptible(pgdat->kcompressd_wait,
				!kfifo_is_empty(&pgdat->kcompressd_fifo));

		for (;;) {
			n = kfifo_out_locked(&pgdat->kcompressd_fifo,
				batch,
				kcompressd_drain_depth() * sizeof(*batch),
				&pgdat->kcompressd_fifo_lock)
					/ sizeof(*batch);
			if (!n)
				break;
			/*
			 * kcompressd is async reclaim writeback; pass a NULL
			 * swap_plug so do_swapout_batch's per-folio fallback
			 * (do_swapout) submits each folio's bio immediately
			 * rather than batching it on a plug the caller would
			 * have to unplug.
			 */
			do_swapout_batch(batch, n, NULL);
		}
	}
	return 0;
}
#endif /* CONFIG_LRU_MARIE */

static inline void count_swpout_vm_event(struct folio *folio)
{
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
	if (unlikely(folio_test_pmd_mappable(folio))) {
		count_memcg_folio_events(folio, THP_SWPOUT, 1);
		count_vm_event(THP_SWPOUT);
	}
#endif
	count_mthp_stat(folio_order(folio), MTHP_STAT_SWPOUT);
	count_memcg_folio_events(folio, PSWPOUT, folio_nr_pages(folio));
	count_vm_events(PSWPOUT, folio_nr_pages(folio));
}

#if defined(CONFIG_MEMCG) && defined(CONFIG_BLK_CGROUP)
static void bio_associate_blkg_from_page(struct bio *bio, struct folio *folio)
{
	struct cgroup_subsys_state *css;
	struct mem_cgroup *memcg;

	memcg = folio_memcg(folio);
	if (!memcg)
		return;

	rcu_read_lock();
	css = cgroup_e_css(memcg->css.cgroup, &io_cgrp_subsys);
	bio_associate_blkg_from_css(bio, css);
	rcu_read_unlock();
}
#else
#define bio_associate_blkg_from_page(bio, folio)		do { } while (0)
#endif /* CONFIG_MEMCG && CONFIG_BLK_CGROUP */

struct swap_iocb {
	struct kiocb		iocb;
	struct bio_vec		bvec[SWAP_CLUSTER_MAX];
	int			pages;
	int			len;
};
static mempool_t *sio_pool;

int sio_pool_init(void)
{
	if (!sio_pool) {
		mempool_t *pool = mempool_create_kmalloc_pool(
			SWAP_CLUSTER_MAX, sizeof(struct swap_iocb));
		if (cmpxchg(&sio_pool, NULL, pool))
			mempool_destroy(pool);
	}
	if (!sio_pool)
		return -ENOMEM;
	return 0;
}

static void sio_write_complete(struct kiocb *iocb, long ret)
{
	struct swap_iocb *sio = container_of(iocb, struct swap_iocb, iocb);
	struct page *page = sio->bvec[0].bv_page;
	int p;

	if (ret != sio->len) {
		/*
		 * In the case of swap-over-nfs, this can be a
		 * temporary failure if the system has limited
		 * memory for allocating transmit buffers.
		 * Mark the page dirty and avoid
		 * folio_rotate_reclaimable but rate-limit the
		 * messages.
		 */
		pr_err_ratelimited("Write error %ld on dio swapfile (%llu)\n",
				   ret, swap_dev_pos(page_swap_entry(page)));
		for (p = 0; p < sio->pages; p++) {
			page = sio->bvec[p].bv_page;
			set_page_dirty(page);
			ClearPageReclaim(page);
		}
	}

	for (p = 0; p < sio->pages; p++)
		end_page_writeback(sio->bvec[p].bv_page);

	mempool_free(sio, sio_pool);
}

static void swap_writepage_fs(struct folio *folio, struct swap_iocb **swap_plug)
{
	struct swap_iocb *sio = swap_plug ? *swap_plug : NULL;
	struct swap_info_struct *sis = __swap_entry_to_info(folio->swap);
	struct file *swap_file = sis->swap_file;
	loff_t pos = swap_dev_pos(folio->swap);

	count_swpout_vm_event(folio);
	folio_start_writeback(folio);
	folio_unlock(folio);
	if (sio) {
		if (sio->iocb.ki_filp != swap_file ||
		    sio->iocb.ki_pos + sio->len != pos) {
			swap_write_unplug(sio);
			sio = NULL;
		}
	}
	if (!sio) {
		sio = mempool_alloc(sio_pool, GFP_NOIO);
		init_sync_kiocb(&sio->iocb, swap_file);
		sio->iocb.ki_complete = sio_write_complete;
		sio->iocb.ki_pos = pos;
		sio->pages = 0;
		sio->len = 0;
	}
	bvec_set_folio(&sio->bvec[sio->pages], folio, folio_size(folio), 0);
	sio->len += folio_size(folio);
	sio->pages += 1;
	if (sio->pages == ARRAY_SIZE(sio->bvec) || !swap_plug) {
		swap_write_unplug(sio);
		sio = NULL;
	}
	if (swap_plug)
		*swap_plug = sio;
}

static void swap_writepage_bdev_sync(struct folio *folio,
		struct swap_info_struct *sis)
{
	struct bio_vec bv;
	struct bio bio;

	bio_init(&bio, sis->bdev, &bv, 1, REQ_OP_WRITE | REQ_SWAP);
	bio.bi_iter.bi_sector = swap_folio_sector(folio);
	bio_add_folio_nofail(&bio, folio, folio_size(folio), 0);

	bio_associate_blkg_from_page(&bio, folio);
	count_swpout_vm_event(folio);

	folio_start_writeback(folio);
	folio_unlock(folio);

	submit_bio_wait(&bio);
	__end_swap_bio_write(&bio);
}

#ifdef CONFIG_LRU_MARIE
/*
 * Stage 1: coalesce a run of folios with *physically contiguous* swap slots
 * into one multi-segment bio and submit_bio_wait() it once. Non-contiguous
 * folios start a new bio (graceful fallback to the per-folio granularity).
 */
static void swap_writepage_bdev_sync_batch(struct folio **folios,
		unsigned int n, struct swap_info_struct *sis)
{
	unsigned int i = 0;

	while (i < n) {
		struct bio *bio;
		sector_t next = swap_folio_sector(folios[i]);
		unsigned int start = i, segs = 0;

		/* Grow the contiguous run [start, i). */
		while (i < n && swap_folio_sector(folios[i]) == next &&
		       segs < BIO_MAX_VECS) {
			next += (folio_nr_pages(folios[i]) << (PAGE_SHIFT - 9));
			segs++;
			i++;
		}

		bio = bio_alloc(sis->bdev, segs,
				REQ_OP_WRITE | REQ_SWAP,
				GFP_NOIO);
		bio->bi_iter.bi_sector = swap_folio_sector(folios[start]);
		bio_associate_blkg_from_page(bio, folios[start]);
		for (unsigned int j = start; j < i; j++) {
			count_swpout_vm_event(folios[j]);
			folio_start_writeback(folios[j]);
			folio_unlock(folios[j]);
			bio_add_folio_nofail(bio, folios[j],
					     folio_size(folios[j]), 0);
		}
		submit_bio_wait(bio);
		__end_swap_bio_write_batch(bio);
		bio_put(bio);
	}
}
#endif /* CONFIG_LRU_MARIE */

static void swap_writepage_bdev_async(struct folio *folio,
		struct swap_info_struct *sis)
{
	struct bio *bio;

	bio = bio_alloc(sis->bdev, 1, REQ_OP_WRITE | REQ_SWAP, GFP_NOIO);
	bio->bi_iter.bi_sector = swap_folio_sector(folio);
	bio->bi_end_io = end_swap_bio_write;
	bio_add_folio_nofail(bio, folio, folio_size(folio), 0);

	bio_associate_blkg_from_page(bio, folio);
	count_swpout_vm_event(folio);
	folio_start_writeback(folio);
	folio_unlock(folio);
	submit_bio(bio);
}

void __swap_writepage(struct folio *folio, struct swap_iocb **swap_plug)
{
	struct swap_info_struct *sis = __swap_entry_to_info(folio->swap);

	VM_BUG_ON_FOLIO(!folio_test_swapcache(folio), folio);
	/*
	 * ->flags can be updated non-atomicially (scan_swap_map_slots),
	 * but that will never affect SWP_FS_OPS, so the data_race
	 * is safe.
	 */
	if (data_race(sis->flags & SWP_FS_OPS))
		swap_writepage_fs(folio, swap_plug);
	/*
	 * ->flags can be updated non-atomicially (scan_swap_map_slots),
	 * but that will never affect SWP_SYNCHRONOUS_IO, so the data_race
	 * is safe.
	 */
	else if (data_race(sis->flags & SWP_SYNCHRONOUS_IO))
		swap_writepage_bdev_sync(folio, sis);
	else
		swap_writepage_bdev_async(folio, sis);
}

void swap_write_unplug(struct swap_iocb *sio)
{
	struct iov_iter from;
	struct address_space *mapping = sio->iocb.ki_filp->f_mapping;
	int ret;

	iov_iter_bvec(&from, ITER_SOURCE, sio->bvec, sio->pages, sio->len);
	ret = mapping->a_ops->swap_rw(&sio->iocb, &from);
	if (ret != -EIOCBQUEUED)
		sio_write_complete(&sio->iocb, ret);
}

static void sio_read_complete(struct kiocb *iocb, long ret)
{
	struct swap_iocb *sio = container_of(iocb, struct swap_iocb, iocb);
	int p;

	if (ret == sio->len) {
		for (p = 0; p < sio->pages; p++) {
			struct folio *folio = page_folio(sio->bvec[p].bv_page);

			count_mthp_stat(folio_order(folio), MTHP_STAT_SWPIN);
			count_memcg_folio_events(folio, PSWPIN, folio_nr_pages(folio));
			folio_mark_uptodate(folio);
			folio_unlock(folio);
		}
		count_vm_events(PSWPIN, sio->pages);
	} else {
		for (p = 0; p < sio->pages; p++) {
			struct folio *folio = page_folio(sio->bvec[p].bv_page);

			folio_unlock(folio);
		}
		pr_alert_ratelimited("Read-error on swap-device\n");
	}
	mempool_free(sio, sio_pool);
}

static bool swap_read_folio_zeromap(struct folio *folio)
{
	int nr_pages = folio_nr_pages(folio);
	struct obj_cgroup *objcg;
	bool is_zeromap;

	/*
	 * Swapping in a large folio that is partially in the zeromap is not
	 * currently handled. Return true without marking the folio uptodate so
	 * that an IO error is emitted (e.g. do_swap_page() will sigbus).
	 */
	if (WARN_ON_ONCE(swap_zeromap_batch(folio->swap, nr_pages,
			&is_zeromap) != nr_pages))
		return true;

	if (!is_zeromap)
		return false;

	objcg = get_obj_cgroup_from_folio(folio);
	count_vm_events(SWPIN_ZERO, nr_pages);
	if (objcg) {
		count_objcg_events(objcg, SWPIN_ZERO, nr_pages);
		obj_cgroup_put(objcg);
	}

	folio_zero_range(folio, 0, folio_size(folio));
	folio_mark_uptodate(folio);
	return true;
}

static void swap_read_folio_fs(struct folio *folio, struct swap_iocb **plug)
{
	struct swap_info_struct *sis = __swap_entry_to_info(folio->swap);
	struct swap_iocb *sio = NULL;
	loff_t pos = swap_dev_pos(folio->swap);

	if (plug)
		sio = *plug;
	if (sio) {
		if (sio->iocb.ki_filp != sis->swap_file ||
		    sio->iocb.ki_pos + sio->len != pos) {
			swap_read_unplug(sio);
			sio = NULL;
		}
	}
	if (!sio) {
		sio = mempool_alloc(sio_pool, GFP_KERNEL);
		init_sync_kiocb(&sio->iocb, sis->swap_file);
		sio->iocb.ki_pos = pos;
		sio->iocb.ki_complete = sio_read_complete;
		sio->pages = 0;
		sio->len = 0;
	}
	bvec_set_folio(&sio->bvec[sio->pages], folio, folio_size(folio), 0);
	sio->len += folio_size(folio);
	sio->pages += 1;
	if (sio->pages == ARRAY_SIZE(sio->bvec) || !plug) {
		swap_read_unplug(sio);
		sio = NULL;
	}
	if (plug)
		*plug = sio;
}

static void swap_read_folio_bdev_sync(struct folio *folio,
		struct swap_info_struct *sis)
{
	struct bio_vec bv;
	struct bio bio;

	bio_init(&bio, sis->bdev, &bv, 1, REQ_OP_READ);
	bio.bi_iter.bi_sector = swap_folio_sector(folio);
	bio_add_folio_nofail(&bio, folio, folio_size(folio), 0);
	/*
	 * Keep this task valid during swap readpage because the oom killer may
	 * attempt to access it in the page fault retry time check.
	 */
	get_task_struct(current);
	count_mthp_stat(folio_order(folio), MTHP_STAT_SWPIN);
	count_memcg_folio_events(folio, PSWPIN, folio_nr_pages(folio));
	count_vm_events(PSWPIN, folio_nr_pages(folio));
	submit_bio_wait(&bio);
	__end_swap_bio_read(&bio);
	put_task_struct(current);
}

static void swap_read_folio_bdev_async(struct folio *folio,
		struct swap_info_struct *sis)
{
	struct bio *bio;

	bio = bio_alloc(sis->bdev, 1, REQ_OP_READ, GFP_KERNEL);
	bio->bi_iter.bi_sector = swap_folio_sector(folio);
	bio->bi_end_io = end_swap_bio_read;
	bio_add_folio_nofail(bio, folio, folio_size(folio), 0);
	count_mthp_stat(folio_order(folio), MTHP_STAT_SWPIN);
	count_memcg_folio_events(folio, PSWPIN, folio_nr_pages(folio));
	count_vm_events(PSWPIN, folio_nr_pages(folio));
	submit_bio(bio);
}

void swap_read_folio(struct folio *folio, struct swap_iocb **plug)
{
	struct swap_info_struct *sis = __swap_entry_to_info(folio->swap);
	bool synchronous = sis->flags & SWP_SYNCHRONOUS_IO;
	bool workingset = folio_test_workingset(folio);
	unsigned long pflags;
	bool in_thrashing;

	VM_BUG_ON_FOLIO(!folio_test_swapcache(folio) && !synchronous, folio);
	VM_BUG_ON_FOLIO(!folio_test_locked(folio), folio);
	VM_BUG_ON_FOLIO(folio_test_uptodate(folio), folio);

	/*
	 * Count submission time as memory stall and delay. When the device
	 * is congested, or the submitting cgroup IO-throttled, submission
	 * can be a significant part of overall IO time.
	 */
	if (workingset) {
		delayacct_thrashing_start(&in_thrashing);
		psi_memstall_enter(&pflags);
	}
	delayacct_swapin_start();

	if (swap_read_folio_zeromap(folio)) {
		folio_unlock(folio);
		goto finish;
	}

	if (zswap_load(folio) != -ENOENT)
		goto finish;

	/* We have to read from slower devices. Increase zswap protection. */
	zswap_folio_swapin(folio);

	if (data_race(sis->flags & SWP_FS_OPS)) {
		swap_read_folio_fs(folio, plug);
	} else if (synchronous) {
		swap_read_folio_bdev_sync(folio, sis);
	} else {
		swap_read_folio_bdev_async(folio, sis);
	}

finish:
	if (workingset) {
		delayacct_thrashing_end(&in_thrashing);
		psi_memstall_leave(&pflags);
	}
	delayacct_swapin_end();
}

void __swap_read_unplug(struct swap_iocb *sio)
{
	struct iov_iter from;
	struct address_space *mapping = sio->iocb.ki_filp->f_mapping;
	int ret;

	iov_iter_bvec(&from, ITER_DEST, sio->bvec, sio->pages, sio->len);
	ret = mapping->a_ops->swap_rw(&sio->iocb, &from);
	if (ret != -EIOCBQUEUED)
		sio_read_complete(&sio->iocb, ret);
}
