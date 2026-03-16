#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
CuTeDSL SM80 (Ampere) implementation of Batched Matrix Multiply (BMM).

ALL LAYOUTS NOW WORKING! ✓

This module provides separate kernel classes for each layout combination:
- BmmRcrKernel: A[M,K] row-major, B[N,K] col-major -> NO TRANSPOSE NEEDED ✓
- BmmRrrKernel: A[M,K] row-major, B[K,N] row-major -> WORKING ✓
- BmmCcrKernel: A[K,M] col-major, B[N,K] col-major -> WORKING ✓

The key fix was correcting the tensor descriptor format in the wrapper:
  - dynamic_shapes[0] = B_dim (batch size)
  - dynamic_shapes[1] = first inner dimension extent
  - dynamic_strides[0] = batch stride

MMA requirements:
  - A operand: [M, K] with K contiguous
  - B operand: [N, K] with K contiguous

Layout analysis:
  - RCR: A[M,K] K-contiguous ✓, B[N,K] K-contiguous ✓ -> IDEAL, no transpose!
  - RRR: A[M,K] K-contiguous ✓, B[K,N] N-contiguous -> logical transpose
  - CCR: A[K,M] M-contiguous -> logical transpose, B[N,K] K-contiguous ✓

Grid: (ceil(M/tile_m), ceil(N/tile_n), B)
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


# =============================================================================
# SM80 helpers
# =============================================================================


def get_smem_layout_atom(
    dtype: Type[cutlass.Numeric], k_dim: int
) -> cute.ComposedLayout:
    """Compute optimal swizzled SMEM layout for SM80."""
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        cutlass.const_expr(
            128
            if bytes_per_row % 128 == 0
            else (
                64
                if bytes_per_row % 64 == 0
                else (32 if bytes_per_row % 32 == 0 else 16)
            )
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if cutlass.const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


# =============================================================================
# BMM RRR Kernel - A[M,K] row-major, B[K,N] row-major
# Following B2B BMM V-tensor transpose pattern
# =============================================================================


class BmmRrrKernel:
    """CuTeDSL SM80 BMM kernel for RRR layout.

    C[B, M, N] = A[B, M, K] @ B[B, K, N] (+ D if has_d)

    A is row-major [M, K] with K contiguous - works for MMA A operand.
    B is row-major [K, N] with N contiguous - needs transpose to [N, K].

    The transpose follows B2B BMM's V tensor pattern exactly:
    1. Load B to SMEM as [K, N] - rows=K, cols=N
    2. Use composition to create [N, K] logical view
    3. Use LdMatrix(transpose=True) for SMEM->register
    """

    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 32,
        has_d: bool = False,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.has_d = has_d

        self.num_threads = 128
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

        # SMEM shapes
        self.sA_shape = (self.tile_m, self.tile_k)
        # B is stored as [K, N] in SMEM, same as GMEM layout
        self.sB_shape = (self.tile_k, self.tile_n)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mD: cute.Tensor,
        B_dim: cutlass.Int32,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch."""
        self._dtype: Type[cutlass.Numeric] = mA.element_type

        # SMEM layout for A: [M, K] with K as the fast dimension
        sA_layout_atom = get_smem_layout_atom(self._dtype, self.tile_k)
        sA_layout = cute.tile_to_shape(sA_layout_atom, self.sA_shape, (0, 1))

        # SMEM layout for B: [K, N] with N as the fast dimension
        # This matches B2B BMM's V layout where the second dimension is fast
        sB_layout_atom = get_smem_layout_atom(self._dtype, self.tile_n)
        sB_layout = cute.tile_to_shape(sB_layout_atom, self.sB_shape, (0, 1))

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sA_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sB_layout)], 1024
            ]

        universal_copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        async_copy_elems = universal_copy_bits // self._dtype.width

        # Tiled copy for A [M, K]
        tA_shape_dim_1 = sA_layout_atom.outer.shape[1] // async_copy_elems
        tA_layout = cute.make_layout(
            (self.num_threads // tA_shape_dim_1, tA_shape_dim_1),
            stride=(tA_shape_dim_1, 1),
        )
        vA_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_A = cute.make_tiled_copy_tv(
            atom_async_copy, tA_layout, vA_layout
        )

        # Tiled copy for B [K, N]
        tB_shape_dim_1 = sB_layout_atom.outer.shape[1] // async_copy_elems
        tB_layout = cute.make_layout(
            (self.num_threads // tB_shape_dim_1, tB_shape_dim_1),
            stride=(tB_shape_dim_1, 1),
        )
        vB_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_B = cute.make_tiled_copy_tv(
            atom_async_copy, tB_layout, vB_layout
        )

        # Tiled MMA
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )

        m_blocks = cute.ceil_div(M, self.tile_m)
        n_blocks = cute.ceil_div(N, self.tile_n)
        grid_dim = (m_blocks, n_blocks, B_dim)

        self.kernel(
            mA,
            mB,
            mC,
            mD,
            M,
            N,
            K,
            sA_layout,
            sB_layout,
            gmem_tiled_copy_A,
            gmem_tiled_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mD: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        gmem_tiled_copy_A: cute.TiledCopy,
        gmem_tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Device kernel for RRR layout."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, n_block, batch_idx = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)

        # Tile GMEM tensors
        gA = cute.local_tile(mA[batch_idx, None, None], self.sA_shape, (m_block, None))
        gB = cute.local_tile(mB[batch_idx, None, None], self.sB_shape, (None, n_block))
        gC = cute.local_tile(
            mC[batch_idx, None, None], (self.tile_m, self.tile_n), (m_block, n_block)
        )

        # Create transposed view for B: [K, N] -> [N, K]
        # Following B2B BMM V transpose pattern exactly:
        # sB is (tile_k, tile_n), we need (tile_n, tile_k) for B-operand
        sB_t = cute.composition(
            sB,
            cute.make_layout(
                (self.tile_n, self.tile_k),
                stride=(self.tile_k, 1),
            ),
        )

        thr_mma = tiled_mma.get_slice(tidx)

        # SMEM copy atoms
        smem_copy_atom_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        # Use transpose=True for B, matching B2B BMM V pattern
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )

        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # Register fragments - use transposed view for B
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB_t))

        tCsA = smem_thr_copy_A.partition_S(sA)
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCsB = smem_thr_copy_B.partition_S(sB_t)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

        acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc.fill(0.0)

        gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(tidx)
        tAgA = gmem_thr_copy_A.partition_S(gA)
        tAsA = gmem_thr_copy_A.partition_D(sA)

        gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(tidx)
        tBgB = gmem_thr_copy_B.partition_S(gB)
        tBsB = gmem_thr_copy_B.partition_D(sB)

        k_tiles = cute.size(gA, mode=[2])

        # Mainloop
        for k_tile in range(k_tiles):
            cute.copy(gmem_tiled_copy_A, tAgA[None, None, None, k_tile], tAsA)
            cute.copy(gmem_tiled_copy_B, tBgB[None, None, None, k_tile], tBsB)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            cute.copy(
                smem_tiled_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0]
            )
            cute.copy(
                smem_tiled_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0]
            )

            for k_block in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                k_next = (k_block + 1) % cute.size(tCsA.shape[2])
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA[None, None, k_next],
                    tCrA_copy_view[None, None, k_next],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB[None, None, k_next],
                    tCrB_copy_view[None, None, k_next],
                )
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    acc,
                )

            self.cta_sync_barrier.arrive_and_wait()

        # Epilogue
        rOut = cute.make_fragment_like(acc, self._dtype)
        rOut.store(acc.load().to(self._dtype))

        if cutlass.const_expr(self.has_d):
            gD = cute.local_tile(
                mD[batch_idx, None, None],
                (self.tile_m, self.tile_n),
                (m_block, n_block),
            )
            tCgD = thr_mma.partition_C(gD)
            for i in cutlass.range_constexpr(cute.size(rOut)):
                d_val = cutlass.Float32(tCgD[i])
                out_val = cutlass.Float32(rOut[i])
                rOut[i] = self._dtype(out_val + d_val)

        tCgC = thr_mma.partition_C(gC)
        for i in cutlass.range_constexpr(cute.size(rOut)):
            tCgC[i] = rOut[i]


# =============================================================================
# BMM CCR Kernel - A[K,M] col-major, B[N,K] col-major
# Following B2B BMM V-tensor transpose pattern
# =============================================================================


class BmmCcrKernel:
    """CuTeDSL SM80 BMM kernel for CCR layout.

    C[B, M, N] = A[B, K, M]^T @ B[B, N, K]^T (+ D if has_d)

    A is col-major [K, M] with M contiguous - needs transpose to [M, K].
    B is col-major [N, K] with K contiguous - works for MMA B operand.

    The transpose follows B2B BMM's V tensor pattern exactly:
    1. Load A to SMEM as [K, M] - rows=K, cols=M
    2. Use composition to create [M, K] logical view
    3. Use LdMatrix(transpose=True) for SMEM->register
    """

    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 32,
        has_d: bool = False,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.has_d = has_d

        self.num_threads = 128
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

        # SMEM shapes
        # A is stored as [K, M] in SMEM, same as GMEM layout
        self.sA_shape = (self.tile_k, self.tile_m)
        self.sB_shape = (self.tile_n, self.tile_k)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mD: cute.Tensor,
        B_dim: cutlass.Int32,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch."""
        self._dtype: Type[cutlass.Numeric] = mA.element_type

        # SMEM layout for A: [K, M] with M as the fast dimension
        sA_layout_atom = get_smem_layout_atom(self._dtype, self.tile_m)
        sA_layout = cute.tile_to_shape(sA_layout_atom, self.sA_shape, (0, 1))

        # SMEM layout for B: [N, K] with K as the fast dimension
        sB_layout_atom = get_smem_layout_atom(self._dtype, self.tile_k)
        sB_layout = cute.tile_to_shape(sB_layout_atom, self.sB_shape, (0, 1))

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sA_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sB_layout)], 1024
            ]

        universal_copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        async_copy_elems = universal_copy_bits // self._dtype.width

        # Tiled copy for A [K, M]
        tA_shape_dim_1 = sA_layout_atom.outer.shape[1] // async_copy_elems
        tA_layout = cute.make_layout(
            (self.num_threads // tA_shape_dim_1, tA_shape_dim_1),
            stride=(tA_shape_dim_1, 1),
        )
        vA_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_A = cute.make_tiled_copy_tv(
            atom_async_copy, tA_layout, vA_layout
        )

        # Tiled copy for B [N, K]
        tB_shape_dim_1 = sB_layout_atom.outer.shape[1] // async_copy_elems
        tB_layout = cute.make_layout(
            (self.num_threads // tB_shape_dim_1, tB_shape_dim_1),
            stride=(tB_shape_dim_1, 1),
        )
        vB_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_B = cute.make_tiled_copy_tv(
            atom_async_copy, tB_layout, vB_layout
        )

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )

        m_blocks = cute.ceil_div(M, self.tile_m)
        n_blocks = cute.ceil_div(N, self.tile_n)
        grid_dim = (m_blocks, n_blocks, B_dim)

        self.kernel(
            mA,
            mB,
            mC,
            mD,
            M,
            N,
            K,
            sA_layout,
            sB_layout,
            gmem_tiled_copy_A,
            gmem_tiled_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mD: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        gmem_tiled_copy_A: cute.TiledCopy,
        gmem_tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Device kernel for CCR layout."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, n_block, batch_idx = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)

        # Tile GMEM tensors
        # A[B, K, M] -> tile as (tile_k, tile_m)
        gA = cute.local_tile(mA[batch_idx, None, None], self.sA_shape, (None, m_block))
        # B[B, N, K] -> tile as (tile_n, tile_k)
        gB = cute.local_tile(mB[batch_idx, None, None], self.sB_shape, (n_block, None))
        gC = cute.local_tile(
            mC[batch_idx, None, None], (self.tile_m, self.tile_n), (m_block, n_block)
        )

        # Create transposed view for A: [K, M] -> [M, K]
        # Following B2B BMM V transpose pattern exactly:
        # sA is (tile_k, tile_m), we need (tile_m, tile_k) for A-operand
        sA_t = cute.composition(
            sA,
            cute.make_layout(
                (self.tile_m, self.tile_k),
                stride=(self.tile_k, 1),
            ),
        )

        thr_mma = tiled_mma.get_slice(tidx)

        # SMEM copy atoms
        # Use transpose=True for A
        smem_copy_atom_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )

        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # Register fragments - use transposed view for A
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA_t))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))

        tCsA = smem_thr_copy_A.partition_S(sA_t)
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCsB = smem_thr_copy_B.partition_S(sB)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

        acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc.fill(0.0)

        gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(tidx)
        tAgA = gmem_thr_copy_A.partition_S(gA)
        tAsA = gmem_thr_copy_A.partition_D(sA)

        gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(tidx)
        tBgB = gmem_thr_copy_B.partition_S(gB)
        tBsB = gmem_thr_copy_B.partition_D(sB)

        k_tiles = cute.size(gA, mode=[2])

        # Mainloop
        for k_tile in range(k_tiles):
            cute.copy(gmem_tiled_copy_A, tAgA[None, None, None, k_tile], tAsA)
            cute.copy(gmem_tiled_copy_B, tBgB[None, None, None, k_tile], tBsB)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            cute.copy(
                smem_tiled_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0]
            )
            cute.copy(
                smem_tiled_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0]
            )

            for k_block in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                k_next = (k_block + 1) % cute.size(tCsA.shape[2])
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA[None, None, k_next],
                    tCrA_copy_view[None, None, k_next],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB[None, None, k_next],
                    tCrB_copy_view[None, None, k_next],
                )
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    acc,
                )

            self.cta_sync_barrier.arrive_and_wait()

        # Epilogue
        rOut = cute.make_fragment_like(acc, self._dtype)
        rOut.store(acc.load().to(self._dtype))

        if cutlass.const_expr(self.has_d):
            gD = cute.local_tile(
                mD[batch_idx, None, None],
                (self.tile_m, self.tile_n),
                (m_block, n_block),
            )
            tCgD = thr_mma.partition_C(gD)
            for i in cutlass.range_constexpr(cute.size(rOut)):
                d_val = cutlass.Float32(tCgD[i])
                out_val = cutlass.Float32(rOut[i])
                rOut[i] = self._dtype(out_val + d_val)

        tCgC = thr_mma.partition_C(gC)
        for i in cutlass.range_constexpr(cute.size(rOut)):
            tCgC[i] = rOut[i]


# =============================================================================
# Legacy wrapper for backward compatibility
# =============================================================================


class BmmRcrKernel:
    """CuTeDSL SM80 BMM kernel for RCR layout - NO TRANSPOSE NEEDED.

    C[B, M, N] = A[B, M, K] @ B[B, N, K]^T (+ D if has_d)

    A is row-major [M, K] with K contiguous ✓ (matches MMA A operand)
    B is col-major [N, K] with K contiguous ✓ (matches MMA B operand)

    This is the IDEAL layout for CuTeDSL MMA operations because:
    - MMA A operand expects [M, K] with K contiguous -> A provides this
    - MMA B operand expects [N, K] with K contiguous -> B provides this
    - NO transpose needed, direct path to MMA!

    This follows the exact same pattern as GEMM RCR.
    """

    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 32,
        has_d: bool = False,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.has_d = has_d

        self.num_threads = 128
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

        # SMEM shapes - both have tile_k as the fast dimension
        self.sA_shape = (self.tile_m, self.tile_k)
        self.sB_shape = (self.tile_n, self.tile_k)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mD: cute.Tensor,
        B_dim: cutlass.Int32,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch."""
        self._dtype: Type[cutlass.Numeric] = mA.element_type

        # SMEM layout for A: [M, K] with K as the fast dimension
        sA_layout_atom = get_smem_layout_atom(self._dtype, self.tile_k)
        sA_layout = cute.tile_to_shape(sA_layout_atom, self.sA_shape, (0, 1))

        # SMEM layout for B: [N, K] with K as the fast dimension
        sB_layout_atom = get_smem_layout_atom(self._dtype, self.tile_k)
        sB_layout = cute.tile_to_shape(sB_layout_atom, self.sB_shape, (0, 1))

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sA_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sB_layout)], 1024
            ]

        universal_copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        async_copy_elems = universal_copy_bits // self._dtype.width

        # Tiled copy for A [M, K]
        tA_shape_dim_1 = sA_layout_atom.outer.shape[1] // async_copy_elems
        tA_layout = cute.make_layout(
            (self.num_threads // tA_shape_dim_1, tA_shape_dim_1),
            stride=(tA_shape_dim_1, 1),
        )
        vA_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_A = cute.make_tiled_copy_tv(
            atom_async_copy, tA_layout, vA_layout
        )

        # Tiled copy for B [N, K]
        tB_shape_dim_1 = sB_layout_atom.outer.shape[1] // async_copy_elems
        tB_layout = cute.make_layout(
            (self.num_threads // tB_shape_dim_1, tB_shape_dim_1),
            stride=(tB_shape_dim_1, 1),
        )
        vB_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_B = cute.make_tiled_copy_tv(
            atom_async_copy, tB_layout, vB_layout
        )

        # Tiled MMA
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )

        m_blocks = cute.ceil_div(M, self.tile_m)
        n_blocks = cute.ceil_div(N, self.tile_n)
        grid_dim = (m_blocks, n_blocks, B_dim)

        self.kernel(
            mA,
            mB,
            mC,
            mD,
            M,
            N,
            K,
            sA_layout,
            sB_layout,
            gmem_tiled_copy_A,
            gmem_tiled_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mD: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        gmem_tiled_copy_A: cute.TiledCopy,
        gmem_tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Device kernel for RCR layout - direct path to MMA, no transpose."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, n_block, batch_idx = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)

        # Tile GMEM tensors
        # A[B, M, K] -> tile as (tile_m, tile_k)
        gA = cute.local_tile(mA[batch_idx, None, None], self.sA_shape, (m_block, None))
        # B[B, N, K] -> tile as (tile_n, tile_k)
        gB = cute.local_tile(mB[batch_idx, None, None], self.sB_shape, (n_block, None))
        gC = cute.local_tile(
            mC[batch_idx, None, None], (self.tile_m, self.tile_n), (m_block, n_block)
        )

        thr_mma = tiled_mma.get_slice(tidx)

        # SMEM copy atoms - NO TRANSPOSE NEEDED
        smem_copy_atom_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )

        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # Register fragments - direct path, no transpose
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))

        tCsA = smem_thr_copy_A.partition_S(sA)
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCsB = smem_thr_copy_B.partition_S(sB)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

        acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc.fill(0.0)

        gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(tidx)
        tAgA = gmem_thr_copy_A.partition_S(gA)
        tAsA = gmem_thr_copy_A.partition_D(sA)

        gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(tidx)
        tBgB = gmem_thr_copy_B.partition_S(gB)
        tBsB = gmem_thr_copy_B.partition_D(sB)

        k_tiles = cute.size(gA, mode=[2])

        # Mainloop
        for k_tile in range(k_tiles):
            cute.copy(gmem_tiled_copy_A, tAgA[None, None, None, k_tile], tAsA)
            cute.copy(gmem_tiled_copy_B, tBgB[None, None, None, k_tile], tBsB)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            cute.copy(
                smem_tiled_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0]
            )
            cute.copy(
                smem_tiled_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0]
            )

            for k_block in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                k_next = (k_block + 1) % cute.size(tCsA.shape[2])
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA[None, None, k_next],
                    tCrA_copy_view[None, None, k_next],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB[None, None, k_next],
                    tCrB_copy_view[None, None, k_next],
                )
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    acc,
                )

            self.cta_sync_barrier.arrive_and_wait()

        # Epilogue
        rOut = cute.make_fragment_like(acc, self._dtype)
        rOut.store(acc.load().to(self._dtype))

        if cutlass.const_expr(self.has_d):
            gD = cute.local_tile(
                mD[batch_idx, None, None],
                (self.tile_m, self.tile_n),
                (m_block, n_block),
            )
            tCgD = thr_mma.partition_C(gD)
            for i in cutlass.range_constexpr(cute.size(rOut)):
                d_val = cutlass.Float32(tCgD[i])
                out_val = cutlass.Float32(rOut[i])
                rOut[i] = self._dtype(out_val + d_val)

        tCgC = thr_mma.partition_C(gC)
        for i in cutlass.range_constexpr(cute.size(rOut)):
            tCgC[i] = rOut[i]


class BmmSm80Kernel:
    """Legacy wrapper that delegates to the appropriate layout-specific kernel."""

    def __new__(
        cls,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 32,
        a_row_major: bool = True,
        b_row_major: bool = True,
        has_d: bool = False,
    ):
        if a_row_major and not b_row_major:
            # RCR layout: A[M,K] row-major, B[N,K] col-major - IDEAL, no transpose
            return BmmRcrKernel(tile_m, tile_n, tile_k, has_d)
        elif a_row_major and b_row_major:
            return BmmRrrKernel(tile_m, tile_n, tile_k, has_d)
        elif not a_row_major and not b_row_major:
            return BmmCcrKernel(tile_m, tile_n, tile_k, has_d)
        else:
            raise NotImplementedError(
                f"Layout combination a_row_major={a_row_major}, "
                f"b_row_major={b_row_major} not yet implemented."
            )
