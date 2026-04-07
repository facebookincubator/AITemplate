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
CuTeDSL SM80 (Ampere) implementation of GEMM RCR (no bias).

Computes: C[M, N] = A[M, K] @ B[N, K]^T

Layout convention (RCR):
  A: Row-major [M, K] (K contiguous) -- K-major
  B: Column-major, stored as [N, K] row-major (K contiguous) -- K-major
  C: Row-major [M, N] (N contiguous)

This is equivalent to torch.nn.functional.linear(A, B) (no bias).

Both A and B are K-major, so we use the TN MMA atom:
  SM80_16x8x16_F16F16F32F32_TN
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from aitemplate.backend.cuda.gemm_universal.cutedsl_gemm_rcr_bias_sm80 import (
    get_smem_layout_atom,
)
from cutlass.cute.nvgpu import cpasync, warp


class GemmRcrSm80Kernel:
    """CuTeDSL SM80 GEMM RCR kernel (no bias).

    C[M, N] = A[M, K] @ B[N, K]^T

    Tile sizes:
      tile_m=128, tile_n=128, tile_k=32
      4 warps (128 threads)
      MMA instruction: 16x8x16

    Grid: (ceil(M/tile_m), ceil(N/tile_n), 1)
    """

    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 32,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.num_threads = 128  # 4 warps

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # [M, K] row-major
        mB: cute.Tensor,  # [N, K] row-major (K contiguous)
        mC: cute.Tensor,  # [M, N] row-major output
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch."""
        self._dtype: Type[cutlass.Numeric] = mA.element_type

        # SMEM layouts for A (tile_m x tile_k) and B (tile_n x tile_k)
        sA_layout_atom = get_smem_layout_atom(self._dtype, self.tile_k)
        sA_layout = cute.tile_to_shape(
            sA_layout_atom,
            (self.tile_m, self.tile_k),
            (0, 1),
        )

        sB_layout_atom = get_smem_layout_atom(self._dtype, self.tile_k)
        sB_layout = cute.tile_to_shape(
            sB_layout_atom,
            (self.tile_n, self.tile_k),
            (0, 1),
        )

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sA_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sB_layout)], 1024
            ]

        # Copy atoms
        universal_copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        # Thread layout for GMEM -> SMEM copies
        async_copy_elems = universal_copy_bits // self._dtype.width
        tAB_shape_dim_1 = sA_layout_atom.outer.shape[1] // async_copy_elems
        tAB_layout = cute.make_layout(
            (self.num_threads // tAB_shape_dim_1, tAB_shape_dim_1),
            stride=(tAB_shape_dim_1, 1),
        )
        vAB_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_AB = cute.make_tiled_copy_tv(
            atom_async_copy, tAB_layout, vAB_layout
        )

        # Tiled copy for output store
        sC_layout_atom = get_smem_layout_atom(self._dtype, self.tile_n)
        sC_layout_atom_outer = sC_layout_atom.outer
        tC_shape_dim_1 = sC_layout_atom_outer.shape[1] // async_copy_elems
        tC_layout = cute.make_layout(
            (self.num_threads // tC_shape_dim_1, tC_shape_dim_1),
            stride=(tC_shape_dim_1, 1),
        )
        vC_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_C = cute.make_tiled_copy_tv(
            atom_universal_copy, tC_layout, vC_layout
        )

        # Tiled MMA (SM80 warp-level)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )

        # Grid: (M_blocks, N_blocks, 1)
        m_blocks = cute.ceil_div(M, self.tile_m)
        n_blocks = cute.ceil_div(N, self.tile_n)
        grid_dim = (m_blocks, n_blocks, 1)

        self.kernel(
            mA,
            mB,
            mC,
            M,
            N,
            K,
            sA_layout,
            sB_layout,
            gmem_tiled_copy_AB,
            gmem_tiled_copy_C,
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
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        gmem_tiled_copy_AB: cute.TiledCopy,
        gmem_tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Device kernel implementing GEMM RCR (no bias)."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, n_block, _ = cute.arch.block_idx()

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)

        # Partition A: tile (tile_m, tile_k) over M and K
        gA = cute.local_tile(
            mA,
            (self.tile_m, self.tile_k),
            (m_block, None),
        )

        # Partition B: tile (tile_n, tile_k) over N and K
        gB = cute.local_tile(
            mB,
            (self.tile_n, self.tile_k),
            (n_block, None),
        )

        # Set up MMA partitions
        thr_mma = tiled_mma.get_slice(tidx)

        # SMEM copy atoms for A and B
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

        # Register fragments
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))

        tCsA = smem_thr_copy_A.partition_S(sA)
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCsB = smem_thr_copy_B.partition_S(sB)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

        # Accumulator for GEMM: C[tile_m, tile_n]
        acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc.fill(0.0)

        # GMEM -> SMEM copy partitions for A
        gmem_thr_copy_AB = gmem_tiled_copy_AB.get_slice(tidx)
        tAgA = gmem_thr_copy_AB.partition_S(gA)
        tAsA = gmem_thr_copy_AB.partition_D(sA)

        # GMEM -> SMEM copy partitions for B
        tBgB = gmem_thr_copy_AB.partition_S(gB)
        tBsB = gmem_thr_copy_AB.partition_D(sB)

        # Number of K-tiles
        k_tiles = cute.size(gA, mode=[2])

        # GEMM mainloop: iterate over K dimension
        for k_tile in range(k_tiles):
            # Load A tile from GMEM to SMEM
            cute.copy(gmem_tiled_copy_AB, tAgA[None, None, None, k_tile], tAsA)
            # Load B tile from GMEM to SMEM
            cute.copy(gmem_tiled_copy_AB, tBgB[None, None, None, k_tile], tBsB)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            # GEMM: acc += A_tile @ B_tile^T
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

        # Epilogue: convert FP32 accumulators to output dtype and store
        rOut = cute.make_fragment_like(acc, self._dtype)
        rOut.store(acc.load().to(self._dtype))

        # Use SMEM for output staging (reuse sA memory)
        sC_layout_atom = get_smem_layout_atom(self._dtype, self.tile_n)
        sC_layout = cute.tile_to_shape(
            sC_layout_atom,
            (self.tile_m, self.tile_n),
            (0, 1),
        )
        sC = cute.make_tensor(sA.iterator, sC_layout)

        smem_copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
        )
        smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
        smem_thr_copy_C = smem_tiled_copy_C.get_slice(tidx)
        taccCrC = smem_thr_copy_C.retile(rOut)
        taccCsC = smem_thr_copy_C.partition_D(sC)
        cute.copy(smem_copy_atom_C, taccCrC, taccCsC)

        self.cta_sync_barrier.arrive_and_wait()

        # Store from SMEM to GMEM
        gC = cute.local_tile(
            mC,
            (self.tile_m, self.tile_n),
            (m_block, n_block),
        )
        gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(tidx)
        tCsC = gmem_thr_copy_C.partition_S(sC)
        tCgC = gmem_thr_copy_C.partition_D(gC)
        tCrC = cute.make_fragment_like(tCgC, self._dtype)
        cute.copy(gmem_tiled_copy_C, tCsC, tCrC)
        cute.copy(gmem_tiled_copy_C, tCrC, tCgC)
