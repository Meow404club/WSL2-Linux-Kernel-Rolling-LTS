# SPDX-License-Identifier: GPL-2.0
#
# Marie LRU. Reached from mm/Makefile via obj-$(CONFIG_LRU_MARIE) += lru_marie/.
# Public dispatch surface for the rest of the kernel is in
# include/linux/lru_marie.h; everything below is Marie-private.

obj-y	+= bitmap.o
obj-y	+= core.o
obj-y	+= state_core.o
obj-y	+= state_reclaim.o
obj-y	+= state_folio.o
obj-y	+= walker.o

obj-$(CONFIG_LRU_MARIE_DEFRAG)	+= defrag.o

ifdef CONFIG_X86
obj-y	+= simd_x86.o
obj-y	+= simd_x86_sse2.o
obj-y	+= simd_x86_avx2.o
obj-y	+= simd_x86_avx512.o
else
obj-y	+= simd_generic.o
endif
