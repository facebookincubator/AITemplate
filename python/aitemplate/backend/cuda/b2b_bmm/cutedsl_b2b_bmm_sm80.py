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
CuTeDSL SM80 (Ampere) implementation of the back-to-back batched GEMM kernel.

Computes: output = activation(alpha0 * Q @ K^T + bias) * alpha1 @ V

GEMM0: score[B,M,N0] = Q[B,M,K0] @ K[B,N0,K0]^T   (Q row-major, K column-major)
Epilogue0: score = activation(alpha0 * score + bias) * alpha1
GEMM1: output[B,M,N1] = score[B,M,N0] @ V[B,N0,N1]  (both row-major)

The key optimization is that GEMM0's output (score) stays in registers and is
used directly as the A-operand for GEMM1, avoiding a round-trip through shared
or global memory.

This mirrors the CUTLASS C++ B2bGemmBatched kernel but written in CuTeDSL.
"""

from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# Activation dispatch
# =============================================================================


def _sigmoid(x):
    return 1.0 / (1.0 + cute.math.exp(-x))


def _relu(x):
    return cute.arch.fmax(x, 0.0)


def _tanh(x):
    return cute.math.tanh(x)


def _silu(x):
    return x * (1.0 / (1.0 + cute.math.exp(-x)))


def _gelu(x):
    return 0.5 * x * (1.0 + cute.math.erf(x * 0.7071067811865476))


def _identity(x):
    return x


ACTIVATION_MAP = {
    "Sigmoid": _sigmoid,
    "ReLu": _relu,
    "Tanh": _tanh,
    "SiLu": _silu,
    "Gelu": _gelu,
    "Identity": _identity,
}


# =============================================================================
# SM80 helpers (mirroring ampere_helpers.py patterns)
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
# B2B BMM SM80 Kernel
# =============================================================================


class B2bBmmSm80Kernel:
    """CuTeDSL SM80 back-to-back batched GEMM kernel.

    Tile sizes match the CUTLASS C++ implementation:
      ThreadblockM=64, ThreadblockK=32, WarpM=16
      InstructionShape = 16x8x16
      N0, N1 are compile-time constants (≤512)

    Parameters
    ----------
    n0 : int
        Inner dimension of GEMM0 output (key sequence length).
    n1 : int
        Output dimension of GEMM1 (value head dim).
    alpha0 : float
        Scale factor applied to GEMM0 output.
    alpha1 : float
        Scale factor applied after activation.
    activation_name : str
        Name of the activation function (Sigmoid, ReLu, Tanh, SiLu, Gelu, Identity).
    has_causal : bool
        Whether to apply causal masking after GEMM0.
    alpha1_divide_by_seq_len : bool
        Whether to divide alpha1 by m0 (sequence length).
    """

    def __init__(
        self,
        n0: int,
        n1: int,
        alpha0: float = 1.0,
        alpha1: float = 1.0,
        activation_name: str = "Identity",
        has_causal: bool = False,
        alpha1_divide_by_seq_len: bool = False,
    ):
        self.n0 = n0
        self.n1 = n1
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.activation_name = activation_name
        self.has_causal = has_causal
        self.alpha1_divide_by_seq_len = alpha1_divide_by_seq_len

        # Tile sizes matching the CUTLASS C++ kernel
        self.threadblock_m = 64
        self.threadblock_k = 32
        self.warp_m = 16
        self.warp_k = 32
        self.num_threads = 128  # 4 warps

        # Named barrier for CTA synchronization
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [batch*heads, M, K0]
        mK: cute.Tensor,  # [batch*heads, N0, K0]
        mV: cute.Tensor,  # [batch*heads, N0, N1]
        mBias: cute.Tensor,  # [batch_or_1, M_or_1, N0] or broadcastable
        mOut: cute.Tensor,  # [batch*heads, M, N1]
        m0: cutlass.Int32,  # sequence length (M dimension)
        k0: cutlass.Int32,  # K dimension
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch."""
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        # SMEM layouts for Q (M x K0), K (N0 x K0), V (N0 x N1)
        sQ_layout_atom = get_smem_layout_atom(self._dtype, self.threadblock_k)
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self.threadblock_m, self.threadblock_k),
            (0, 1),
        )

        sK_layout_atom = get_smem_layout_atom(self._dtype, self.threadblock_k)
        sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            (self.n0, self.threadblock_k),
            (0, 1),
        )

        sV_layout_atom = get_smem_layout_atom(self._dtype, self.n1)
        sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.n0, self.n1),
            (0, 1),
        )

        sBias_layout_atom = get_smem_layout_atom(self._dtype, self.n0)
        sBias_layout = cute.tile_to_shape(
            sBias_layout_atom,
            (self.threadblock_m, self.n0),
            (0, 1),
        )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024
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

        # Thread layout for GMEM copies
        async_copy_elems = universal_copy_bits // self._dtype.width
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQK_layout = cute.make_layout(
            (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            stride=(tQK_shape_dim_1, 1),
        )
        vQK_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_QK = cute.make_tiled_copy_tv(
            atom_async_copy, tQK_layout, vQK_layout
        )

        # Tiled copy for V
        sV_layout_atom_outer = sV_layout_atom.outer
        tV_shape_dim_1 = sV_layout_atom_outer.shape[1] // async_copy_elems
        tV_layout = cute.make_layout(
            (self.num_threads // tV_shape_dim_1, tV_shape_dim_1),
            stride=(tV_shape_dim_1, 1),
        )
        vV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_V = cute.make_tiled_copy_tv(
            atom_async_copy, tV_layout, vV_layout
        )

        # Tiled copy for Bias
        tBias_shape_dim_1 = sBias_layout_atom.outer.shape[1] // async_copy_elems
        tBias_layout = cute.make_layout(
            (self.num_threads // tBias_shape_dim_1, tBias_shape_dim_1),
            stride=(tBias_shape_dim_1, 1),
        )
        vBias_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_Bias = cute.make_tiled_copy_tv(
            atom_async_copy, tBias_layout, vBias_layout
        )

        # Tiled copy for output store
        gmem_tiled_copy_Out = cute.make_tiled_copy_tv(
            atom_universal_copy, tV_layout, vV_layout
        )

        # Tiled MMA (SM80 warp-level)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )

        # Grid: (M_blocks, batch*heads)
        m_blocks = cute.ceil_div(m0, self.threadblock_m)
        grid_dim = (m_blocks, cute.size(mQ.shape[0]), 1)

        self.kernel(
            mQ,
            mK,
            mV,
            mBias,
            mOut,
            m0,
            k0,
            sQ_layout,
            sK_layout,
            sV_layout,
            sBias_layout,
            gmem_tiled_copy_QK,
            gmem_tiled_copy_V,
            gmem_tiled_copy_Bias,
            gmem_tiled_copy_Out,
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
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mBias: cute.Tensor,
        mOut: cute.Tensor,
        m0: cutlass.Int32,
        k0: cutlass.Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sBias_layout: cute.ComposedLayout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_Bias: cute.TiledCopy,
        gmem_tiled_copy_Out: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Device kernel implementing B2B batched GEMM."""
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_head, _ = cute.arch.block_idx()

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sBias = storage.sBias.get_tensor(sBias_layout)

        # =====================================================================
        # GEMM0: score = Q @ K^T
        # Q is [batch*heads, M, K0] row-major
        # K is [batch*heads, N0, K0] column-major (treated as K^T)
        # =====================================================================

        # Partition Q: tile (threadblock_m, threadblock_k) over M and K
        gQ = cute.local_tile(
            mQ[batch_head, None, None],
            (self.threadblock_m, self.threadblock_k),
            (m_block, None),
        )

        # Partition K: tile (N0, threadblock_k) over K (K is loaded fully in N, tiled in K)
        gK = cute.local_tile(
            mK[batch_head, None, None],
            (self.n0, self.threadblock_k),
            (0, None),
        )

        # Set up MMA partitions
        thr_mma = tiled_mma.get_slice(tidx)

        # SMEM copy atoms for GEMM0
        smem_copy_atom_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )

        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)

        # Register fragments for GEMM0
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)

        # Accumulator for GEMM0: score[threadblock_m, N0]
        acc_shape_S = thr_mma.partition_shape_C((self.threadblock_m, self.n0))
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # GMEM -> SMEM copy partitions for Q
        gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
        tQgQ = gmem_thr_copy_QK.partition_S(gQ)
        tQsQ = gmem_thr_copy_QK.partition_D(sQ)

        # GMEM -> SMEM copy partitions for K
        tKgK = gmem_thr_copy_QK.partition_S(gK)
        tKsK = gmem_thr_copy_QK.partition_D(sK)

        # Number of K-tiles for GEMM0
        k_tiles = cute.size(gQ, mode=[2])

        # =====================================================================
        # GEMM0 mainloop: iterate over K dimension
        # =====================================================================
        for k_tile in range(k_tiles):
            # Load Q tile from GMEM to SMEM
            # tQgQ has 4 modes: (CPY_Atom, CPY_M, CPY_K, k_tiles)
            cute.copy(gmem_tiled_copy_QK, tQgQ[None, None, None, k_tile], tQsQ)
            # Load K tile from GMEM to SMEM
            cute.copy(gmem_tiled_copy_QK, tKgK[None, None, None, k_tile], tKsK)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            # GEMM0: score += Q_tile @ K_tile^T
            # Load first k-block from SMEM to registers
            cute.copy(
                smem_tiled_copy_Q, tSsQ[None, None, 0], tSrQ_copy_view[None, None, 0]
            )
            cute.copy(
                smem_tiled_copy_K, tSsK[None, None, 0], tSrK_copy_view[None, None, 0]
            )

            for k_block in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                k_next = (k_block + 1) % cute.size(tSsQ.shape[2])
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k_next],
                    tSrQ_copy_view[None, None, k_next],
                )
                cute.copy(
                    smem_tiled_copy_K,
                    tSsK[None, None, k_next],
                    tSrK_copy_view[None, None, k_next],
                )
                cute.gemm(
                    tiled_mma,
                    acc_S,
                    tSrQ[None, None, k_block],
                    tSrK[None, None, k_block],
                    acc_S,
                )

            self.cta_sync_barrier.arrive_and_wait()

        # =====================================================================
        # Epilogue0: Apply alpha0, add bias, activation, alpha1
        # =====================================================================

        # Load bias from GMEM to SMEM
        # Bias shape: [batch_or_1, M_or_1, N0]
        gBias = cute.local_tile(
            mBias[batch_head % mBias.shape[0], None, None],
            (self.threadblock_m, self.n0),
            (m_block, 0),
        )
        gmem_thr_copy_Bias = gmem_tiled_copy_Bias.get_slice(tidx)
        tBgBias = gmem_thr_copy_Bias.partition_S(gBias)
        tBsBias = gmem_thr_copy_Bias.partition_D(sBias)
        cute.copy(gmem_tiled_copy_Bias, tBgBias, tBsBias)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Partition bias for MMA accumulator layout
        tCsBias = thr_mma.partition_C(sBias)

        # Apply epilogue: scale, bias, activation, alpha1
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        tCsBias_mn = self._make_acc_tensor_mn_view(tCsBias)

        effective_alpha1 = self.alpha1
        if self.alpha1_divide_by_seq_len:
            effective_alpha1 = self.alpha1 / cutlass.Float32(m0)

        activation_fn = ACTIVATION_MAP[self.activation_name]

        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                val = acc_S_mn[r, c]
                # Scale by alpha0
                val = val * cutlass.Float32(self.alpha0)
                # Add bias (loaded from SMEM, cast to f32)
                val = val + cutlass.Float32(tCsBias_mn[r, c])
                # Apply activation
                val = activation_fn(val)
                # Apply causal mask if needed
                if self.has_causal:
                    # TODO: implement causal mask check based on m,n coordinates
                    pass
                # Scale by alpha1
                val = val * cutlass.Float32(effective_alpha1)
                acc_S_mn[r, c] = val

        # =====================================================================
        # GEMM1: output = score @ V
        # score is in registers (acc_S), V is loaded from GMEM to SMEM
        # =====================================================================

        # Load V from GMEM to SMEM
        gV = mV[batch_head, None, None]
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        tVgV = gmem_thr_copy_V.partition_S(gV)
        tVsV = gmem_thr_copy_V.partition_D(sV)
        cute.copy(gmem_tiled_copy_V, tVgV, tVsV)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Transpose V in SMEM: V is (N0, N1), we need (N1, N0) for B-operand
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self.n1, self.n0),
                stride=(self.n0, 1),
            ),
        )

        # SMEM copy atom for V (transposed)
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # Accumulator for GEMM1: output[threadblock_m, N1]
        acc_shape_O = thr_mma.partition_shape_C((self.threadblock_m, self.n1))
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # Convert acc_S (score) to fp16 for use as A-operand in GEMM1
        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))

        # Reshape score to match MMA A-operand layout for GEMM1
        # acc_S is partitioned as (MMA_atom, MMA_M, MMA_N) for GEMM0's (M, N0) output
        # For GEMM1, score is A-operand with shape (M, N0) where N0 is the K-dim
        # Need to convert: (4, MMA_M, MMA_N) -> ((4, 2), MMA_M, MMA_N / 2)
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        # GEMM1 mainloop: score (registers) @ V (SMEM)
        cute.copy(
            smem_tiled_copy_V, tOsVt[None, None, 0], tOrVt_copy_view[None, None, 0]
        )
        for k_block in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k_block + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_tiled_copy_V,
                tOsVt[None, None, k_next],
                tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma,
                acc_O,
                tOrS[None, None, k_block],
                tOrVt[None, None, k_block],
                acc_O,
            )

        # =====================================================================
        # Store output from registers to GMEM
        # =====================================================================
        rOut = cute.make_fragment_like(acc_O, self._dtype)
        rOut.store(acc_O.load().to(self._dtype))

        # Write to SMEM first, then GMEM for better vectorization
        sOut_layout_atom = get_smem_layout_atom(self._dtype, self.n1)
        sOut_layout = cute.tile_to_shape(
            sOut_layout_atom,
            (self.threadblock_m, self.n1),
            (0, 1),
        )
        # Reuse sQ's memory for output staging
        sOut = cute.make_tensor(sQ.iterator, sOut_layout)

        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
        )
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rOut)
        taccOsO = smem_thr_copy_O.partition_D(sOut)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        self.cta_sync_barrier.arrive_and_wait()

        # Store from SMEM to GMEM
        gOut = cute.local_tile(
            mOut[batch_head, None, None],
            (self.threadblock_m, self.n1),
            (m_block, 0),
        )
        gmem_thr_copy_Out = gmem_tiled_copy_Out.get_slice(tidx)
        tOsOut = gmem_thr_copy_Out.partition_S(sOut)
        tOgOut = gmem_thr_copy_Out.partition_D(gOut)
        tOrOut = cute.make_fragment_like(tOgOut, self._dtype)
        cute.copy(gmem_tiled_copy_Out, tOsOut, tOrOut)
        cute.copy(gmem_tiled_copy_Out, tOrOut, tOgOut)

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
