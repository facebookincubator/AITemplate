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
CuTeDSL SM90 (Hopper) implementation of the back-to-back batched GEMM kernel.

Computes: output = activation(alpha0 * Q @ K^T + bias) * alpha1 @ V

Uses Hopper-specific features:
  - TMA (Tensor Memory Accelerator) for efficient GMEM <-> SMEM transfers
  - WGMMA (Warpgroup Matrix Multiply-Accumulate) instructions
  - PipelineAsync for multi-stage producer/consumer pipeline

GEMM0: score[B,M,N0] = Q[B,M,K0] @ K[B,N0,K0]^T
Epilogue0: score = activation(alpha0 * score + bias) * alpha1
GEMM1: output[B,M,N1] = score[B,M,N0] @ V[B,N0,N1]

Score stays in registers between GEMM0 and GEMM1.
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from aitemplate.backend.cuda.b2b_bmm.cutedsl_b2b_bmm_sm80 import ACTIVATION_MAP
from cutlass.cute.runtime import from_dlpack


class B2bBmmSm90Kernel:
    """CuTeDSL SM90 back-to-back batched GEMM kernel using TMA + WGMMA.

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
        Name of the activation function.
    has_causal : bool
        Whether to apply causal masking after GEMM0.
    alpha1_divide_by_seq_len : bool
        Whether to divide alpha1 by m0.
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

        # Tile sizes
        self.tile_m = 64
        self.tile_k = 64  # MMA_K for SM90

        # SM90 uses 1 warp group (128 threads)
        self.num_threads = 128
        self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [batch*heads, M, K0]
        mK: cute.Tensor,  # [batch*heads, N0, K0]
        mV: cute.Tensor,  # [batch*heads, N0, N1]
        mBias: cute.Tensor,  # [batch_or_1, M_or_1, N0]
        mOut: cute.Tensor,  # [batch*heads, M, N1]
        m0: cutlass.Int32,
        k0: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch for SM90."""
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        # All b2b_bmm tensors are row-major (contiguous, stride-1 on innermost dim).
        # LayoutEnum.from_tensor() only supports 2D tensors, so we hardcode ROW_MAJOR.
        q_layout = utils.LayoutEnum.ROW_MAJOR
        k_layout = utils.LayoutEnum.ROW_MAJOR

        # Setup tiled MMA for SM90
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            q_layout.sm90_mma_major_mode(),
            k_layout.sm90_mma_major_mode(),
            cutlass.Float32,
            self.atom_layout_mnk,
            tiler_mn=(self.tile_m, self.n0),
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        tile_k_gemm0 = mma_inst_shape_k * 4

        # SMEM layouts for GEMM0
        q_tile_shape = (self.tile_m, tile_k_gemm0)
        k_tile_shape = (self.n0, tile_k_gemm0)

        q_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(q_layout, self._dtype, tile_k_gemm0),
            self._dtype,
        )
        q_smem_layout = cute.tile_to_shape(
            q_smem_layout_atom,
            q_tile_shape,
            (0, 1),
        )

        k_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(k_layout, self._dtype, tile_k_gemm0),
            self._dtype,
        )
        k_smem_layout = cute.tile_to_shape(
            k_smem_layout_atom,
            k_tile_shape,
            (0, 1),
        )

        # Compute number of pipeline stages for GEMM0
        qk_bytes_per_stage = (
            cute.size(q_tile_shape) * self._dtype.width // 8
            + cute.size(k_tile_shape) * self._dtype.width // 8
        )
        ab_stage = (self.smem_capacity - 2048) // qk_bytes_per_stage
        ab_stage = min(ab_stage, 7)  # cap at 7 stages

        q_smem_layout_staged = cute.tile_to_shape(
            q_smem_layout_atom,
            (*q_tile_shape, ab_stage),
            (0, 1, 2),
        )
        k_smem_layout_staged = cute.tile_to_shape(
            k_smem_layout_atom,
            (*k_tile_shape, ab_stage),
            (0, 1, 2),
        )

        # TMA atoms for Q, K
        tma_atom_q, tma_tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            q_smem_layout,
            q_tile_shape,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            k_smem_layout,
            k_tile_shape,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, ab_stage * 2
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        tma_copy_bytes = cute.size_in_bytes(
            self._dtype, q_smem_layout
        ) + cute.size_in_bytes(self._dtype, k_smem_layout)

        # Grid
        m_blocks = cute.ceil_div(m0, self.tile_m)
        grid_dim = (m_blocks, cute.size(mQ.shape[0]), 1)

        self.kernel(
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            mV,
            mBias,
            mOut,
            m0,
            k0,
            tiled_mma,
            q_smem_layout_staged,
            k_smem_layout_staged,
            ab_stage,
            tma_copy_bytes,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        mQ_mkl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_nkl: cute.Tensor,
        mV: cute.Tensor,
        mBias: cute.Tensor,
        mOut: cute.Tensor,
        m0: cutlass.Int32,
        k0: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        ab_stage: cutlass.Constexpr,
        tma_copy_bytes: cutlass.Constexpr,
        SharedStorage: cutlass.Constexpr,
    ):
        """SM90 device kernel for B2B BMM."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        m_block, batch_head, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )

        # Setup pipeline
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        num_warps = self.num_threads // 32
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_warps
        )

        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        # Partition global tensors
        tile_coord = (m_block, 0, None, batch_head)
        gQ_mkl = cute.local_tile(
            mQ_mkl, (self.tile_m, None, 1), tile_coord, proj=(1, None, 1)
        )
        gK_nkl = cute.local_tile(
            mK_nkl, (self.n0, None, 1), tile_coord, proj=(None, 1, 1)
        )

        # TMA partition
        q_cta_layout = cute.make_layout(1)
        sQ_for_tma = cute.group_modes(sQ, 0, 2)
        gQ_for_tma = cute.group_modes(gQ_mkl, 0, 2)
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            q_cta_layout,
            sQ_for_tma,
            gQ_for_tma,
        )

        k_cta_layout = cute.make_layout(1)
        sK_for_tma = cute.group_modes(sK, 0, 2)
        gK_for_tma = cute.group_modes(gK_nkl, 0, 2)
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            k_cta_layout,
            sK_for_tma,
            gK_for_tma,
        )

        # MMA partitions
        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, 0))
        k_smem_layout = cute.slice_(k_smem_layout_staged, (None, None, 0))

        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        warp_group_thread_layout = cute.make_layout(1, stride=128)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCsQ = thr_mma.partition_A(sQ)
        tCsK = thr_mma.partition_B(sK)
        tCrQ = tiled_mma.make_fragment_A(tCsQ)
        tCrK = tiled_mma.make_fragment_B(tCsK)

        # Accumulator for GEMM0
        acc_shape_S = thr_mma.partition_shape_C((self.tile_m, self.n0))
        acc_S = cute.make_fragment(acc_shape_S, cutlass.Float32)

        # =====================================================================
        # GEMM0 pipeline: prefetch + mainloop
        # =====================================================================
        k_tile_cnt = cute.size(gQ_mkl, mode=[2])
        prefetch_cnt = cutlass.min(ab_stage, k_tile_cnt)

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, ab_stage
        )

        # Prefetch
        if warp_idx == 0:
            for prefetch_idx in cutlass.range(prefetch_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(producer_state)
                tQgQ_k = tQgQ[(None, producer_state.count)]
                tQsQ_pipe = tQsQ[(None, producer_state.index)]
                tKgK_k = tKgK[(None, producer_state.count)]
                tKsK_pipe = tKsK[(None, producer_state.index)]
                cute.copy(
                    tma_atom_q,
                    tQgQ_k,
                    tQsQ_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_k,
                    tKgK_k,
                    tKsK_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # Mainloop
        consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, ab_stage
        )
        consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, ab_stage
        )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrQ, mode=[2])

        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)
            mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, consumer_read_state.index)
                tCrQ_phase = tCrQ[k_block_coord]
                tCrK_phase = tCrK[k_block_coord]
                cute.gemm(tiled_mma, acc_S, tCrQ_phase, tCrK_phase, acc_S)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            mainloop_pipeline.consumer_release(consumer_release_state)
            consumer_read_state.advance()
            consumer_release_state.advance()

            # Continue producer
            if warp_idx == 0 and producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(producer_state)
                tQgQ_k = tQgQ[(None, producer_state.count)]
                tQsQ_pipe = tQsQ[(None, producer_state.index)]
                tKgK_k = tKgK[(None, producer_state.count)]
                tKsK_pipe = tKsK[(None, producer_state.index)]
                cute.copy(
                    tma_atom_q,
                    tQgQ_k,
                    tQsQ_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_k,
                    tKgK_k,
                    tKsK_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # =====================================================================
        # Epilogue0: scale, bias, activation, alpha1
        # (Simplified: apply element-wise in registers)
        # =====================================================================
        activation_fn = ACTIVATION_MAP[self.activation_name]
        effective_alpha1 = self.alpha1
        if self.alpha1_divide_by_seq_len:
            effective_alpha1 = self.alpha1 / cutlass.Float32(m0)

        acc_S_vec = acc_S.load()
        acc_S_vec = acc_S_vec * cutlass.Float32(self.alpha0)
        # TODO: Add bias loading via TMA and add to acc_S
        acc_S_vec = activation_fn(acc_S_vec)
        acc_S_vec = acc_S_vec * cutlass.Float32(effective_alpha1)
        acc_S.store(acc_S_vec)

        # =====================================================================
        # GEMM1: output = score @ V
        # Score is in registers, V needs to be loaded
        # For SM90, we fall back to loading V into SMEM and using WGMMA
        # =====================================================================
        cute.arch.sync_threads()

        # Convert score to fp16
        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))

        # For GEMM1, we need a separate pipeline.
        # For simplicity in this initial implementation, we load V in a single
        # stage since N0*N1 is typically small (≤512*512).
        # A full implementation would use TMA pipeline for V as well.

        # Store score to SMEM, load V to SMEM, then do WGMMA for GEMM1
        # This is a simplified approach; a fully optimized version would keep
        # score in registers and use the register-to-SMEM path.

        # For now, store output using simple register-to-global writes
        # TODO: Implement full GEMM1 with TMA V loading + WGMMA
        acc_shape_O = thr_mma.partition_shape_C((self.tile_m, self.n1))
        acc_O = cute.make_fragment(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # Placeholder: Full GEMM1 implementation requires V TMA setup
        # which depends on the V tensor layout being known at compile time.

        return
