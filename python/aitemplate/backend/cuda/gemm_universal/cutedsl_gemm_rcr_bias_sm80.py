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
CuTeDSL SM80 (Ampere) implementation of GEMM RCR with bias.

Computes: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]

Layout convention (RCR):
  A: Row-major [M, K] (K contiguous) -- K-major
  B: Column-major, stored as [N, K] row-major (K contiguous) -- K-major
  C: Row-major [M, N] (N contiguous)
  Bias: 1D [N], broadcast across all M rows

This is equivalent to torch.nn.functional.linear(A, B, bias=Bias).

Both A and B are K-major, so we use the TN MMA atom:
  SM80_16x8x16_F16F16F32F32_TN
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


# =============================================================================
# SM80 helpers (same as cutedsl_b2b_bmm_sm80.py)
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
# GEMM RCR Bias SM80 Kernel
# =============================================================================


class GemmRcrBiasSm80Kernel:
    """CuTeDSL SM80 GEMM RCR with 1D bias kernel.

    C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]

    Tile sizes:
      tile_m=128, tile_n=128, tile_k=32
      4 warps (128 threads)
      MMA instruction: 16x8x16

    Grid: (ceil(M/tile_m), ceil(N/tile_n), 1)

    The bias is added after the GEMM mainloop using a SMEM-based approach:
    1. Load bias[tile_n] into SMEM as a 1D vector
    2. Create a 2D broadcast view (tile_m, tile_n) with stride-0 in M
    3. Partition to match accumulator layout via thr_mma.partition_C()
    4. Element-wise add in FP32

    Parameters
    ----------
    tile_m : int
        Threadblock tile size in M dimension.
    tile_n : int
        Threadblock tile size in N dimension.
    tile_k : int
        Threadblock tile size in K dimension (loop step).
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

        # Named barrier for CTA synchronization
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # [M, K] row-major
        mB: cute.Tensor,  # [N, K] row-major (K contiguous)
        mBias: cute.Tensor,  # [N] 1D bias
        mC: cute.Tensor,  # [M, N] row-major output
        M: cutlass.Int32,  # M dimension
        N: cutlass.Int32,  # N dimension
        K: cutlass.Int32,  # K dimension
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

        # SMEM layout for bias broadcast: (tile_m, tile_n) with N as inner dim
        # We use a layout atom based on tile_n for the broadcast view
        sBias_layout_atom = get_smem_layout_atom(self._dtype, self.tile_n)
        sBias_layout = cute.tile_to_shape(
            sBias_layout_atom,
            (self.tile_m, self.tile_n),
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
            sBias: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sBias_layout)], 1024
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

        # Thread layout for GMEM -> SMEM copies (A and B use same tile_k)
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

        # Tiled copy for bias (uses sBias_layout_atom for thread mapping)
        tBias_shape_dim_1 = sBias_layout_atom.outer.shape[1] // async_copy_elems
        tBias_layout = cute.make_layout(
            (self.num_threads // tBias_shape_dim_1, tBias_shape_dim_1),
            stride=(tBias_shape_dim_1, 1),
        )
        vBias_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_Bias = cute.make_tiled_copy_tv(
            atom_async_copy, tBias_layout, vBias_layout
        )

        # Tiled copy for output store (C has tile_n as inner dim)
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

        # Create 2D broadcast view of bias: (M, N) with stride (0, bias_stride)
        # mBias is 1D [N], so its stride in mode-0 is the element stride.
        # We broadcast it across M rows by setting M-stride to 0.
        bias_stride = mBias.layout.stride[0]
        mBias_2d = cute.make_tensor(
            mBias.iterator,
            cute.make_layout(
                (cute.size(mA.shape[0]), cute.size(mBias.shape[0])),
                stride=(0, bias_stride),
            ),
        )

        # Grid: (M_blocks, N_blocks, 1)
        m_blocks = cute.ceil_div(M, self.tile_m)
        n_blocks = cute.ceil_div(N, self.tile_n)
        grid_dim = (m_blocks, n_blocks, 1)

        self.kernel(
            mA,
            mB,
            mBias_2d,
            mC,
            M,
            N,
            K,
            sA_layout,
            sB_layout,
            sBias_layout,
            gmem_tiled_copy_AB,
            gmem_tiled_copy_Bias,
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
        mBias: cute.Tensor,
        mC: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sBias_layout: cute.ComposedLayout,
        gmem_tiled_copy_AB: cute.TiledCopy,
        gmem_tiled_copy_Bias: cute.TiledCopy,
        gmem_tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Device kernel implementing GEMM RCR + bias."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, n_block, _ = cute.arch.block_idx()

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)
        sBias = storage.sBias.get_tensor(sBias_layout)

        # =====================================================================
        # GEMM: C = A @ B^T
        # A is [M, K] row-major (K contiguous)
        # B is [N, K] row-major (K contiguous) -- treated as B^T in GEMM
        # Both are K-major -> TN MMA atom
        # =====================================================================

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

        # =====================================================================
        # GEMM mainloop: iterate over K dimension
        # =====================================================================
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

        # =====================================================================
        # Epilogue: add 1D bias via SMEM broadcast
        #
        # Strategy: mBias is already a 2D broadcast view (M, N) with stride-0
        # in M, created in __call__. We tile it by (tile_m, tile_n) and load
        # into SMEM. Then partition_C() aligns it to the accumulator layout.
        # =====================================================================

        # Partition bias: tile (tile_m, tile_n) over M and N
        gBias = cute.local_tile(
            mBias,
            (self.tile_m, self.tile_n),
            (m_block, n_block),
        )

        # Load bias into SMEM using async copy
        gmem_thr_copy_Bias = gmem_tiled_copy_Bias.get_slice(tidx)
        tBiasgBias = gmem_thr_copy_Bias.partition_S(gBias)
        tBiassBias = gmem_thr_copy_Bias.partition_D(sBias)
        cute.copy(gmem_tiled_copy_Bias, tBiasgBias, tBiassBias)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Partition bias SMEM for MMA accumulator layout
        tCsBias = thr_mma.partition_C(sBias)

        # Apply bias: acc += bias (in FP32)
        acc_mn = self._make_acc_tensor_mn_view(acc)
        tCsBias_mn = self._make_acc_tensor_mn_view(tCsBias)

        for r in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                acc_mn[r, c] = acc_mn[r, c] + cutlass.Float32(tCsBias_mn[r, c])

        # =====================================================================
        # Store output from registers to GMEM via SMEM
        # =====================================================================
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

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Convert MMA accumulator to (M, N) logical view."""
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)
