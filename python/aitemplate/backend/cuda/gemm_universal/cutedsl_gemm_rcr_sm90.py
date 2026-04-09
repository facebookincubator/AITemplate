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
CuTeDSL SM90 (Hopper) implementation of GEMM RCR (no bias).

Computes: C[M, N] = A[M, K] @ B[N, K]^T

Layout convention (RCR):
  A: Row-major [M, K] (K contiguous) -- K-major
  B: Column-major, stored as [N, K] row-major (K contiguous) -- K-major
  C: Row-major [M, N] (N contiguous)

Uses Hopper-specific features:
  - TMA (Tensor Memory Accelerator) for efficient GMEM <-> SMEM transfers
  - WGMMA (Warpgroup Matrix Multiply-Accumulate) instructions
  - PipelineTmaAsync for multi-stage producer/consumer pipeline
  - TMA store for epilogue output

This is equivalent to torch.nn.functional.linear(A, B) (no bias).
"""

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils


class GemmRcrSm90Kernel:
    """CuTeDSL SM90 GEMM RCR kernel (no bias) using TMA + WGMMA.

    C[M, N] = A[M, K] @ B[N, K]^T

    Tile sizes:
      tile_m=128, tile_n=128, tile_k=64 (MMA_K for SM90)
      1 warp group (128 threads)

    Grid: (ceil(M/tile_m), ceil(N/tile_n), 1)
    """

    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = 64  # Updated in _setup_attributes

        self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.num_threads = self.mma_warp_groups * self.num_threads_per_warp_group

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

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

        a_layout = utils.LayoutEnum.ROW_MAJOR
        b_layout = utils.LayoutEnum.ROW_MAJOR
        c_layout = utils.LayoutEnum.ROW_MAJOR

        # Setup tiled MMA for SM90
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            cutlass.Float32,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_n),
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        tile_k = mma_inst_shape_k * 4

        # SMEM layouts for A and B
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, self._dtype, tile_k),
            self._dtype,
        )
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, self._dtype, tile_k),
            self._dtype,
        )

        # Compute pipeline stages
        a_smem_shape = (self.tile_m, tile_k)
        b_smem_shape = (self.tile_n, tile_k)
        ab_bytes_per_stage = (
            cute.size(a_smem_shape) * self._dtype.width // 8
            + cute.size(b_smem_shape) * self._dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        ab_stage = (self.smem_capacity - mbar_helpers_bytes) // ab_bytes_per_stage
        ab_stage = min(ab_stage, 7)

        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            (*a_smem_shape, ab_stage),
            (0, 1, 2),
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            (*b_smem_shape, ab_stage),
            (0, 1, 2),
        )

        # Epilogue SMEM layout for output store
        epi_tile = (min(64, self.tile_m), min(32, self.tile_n))
        epi_stage = 4

        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(c_layout, self._dtype, epi_tile[1]),
            self._dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            (*epi_tile, epi_stage),
            (0, 1, 2),
        )

        # TMA atoms for A and B
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mA,
            a_smem_layout,
            a_smem_shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mB,
            b_smem_layout,
            b_smem_shape,
        )

        # TMA atom for C store
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        c_cta_v_layout = cute.composition(cute.make_identity_layout(mC.shape), epi_tile)
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mC,
            epi_smem_layout,
            c_cta_v_layout,
        )

        tma_copy_bytes = cute.size_in_bytes(
            self._dtype, a_smem_layout
        ) + cute.size_in_bytes(self._dtype, b_smem_layout)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        # Grid: (M_blocks, N_blocks, 1)
        m_blocks = cute.ceil_div(M, self.tile_m)
        n_blocks = cute.ceil_div(N, self.tile_n)
        grid_dim = (m_blocks, n_blocks, 1)

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            M,
            N,
            K,
            tiled_mma,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_tile,
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
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cutlass.Constexpr,
        ab_stage: cutlass.Constexpr,
        tma_copy_bytes: cutlass.Constexpr,
        SharedStorage: cutlass.Constexpr,
    ):
        """SM90 device kernel for GEMM RCR (no bias)."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        m_block, n_block, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # Setup pipeline
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_warps = self.num_threads // 32
        mainloop_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_warps
        )
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=ab_stage,
            producer_group=mainloop_producer_group,
            consumer_group=mainloop_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        # Partition global tensors for TMA
        tile_shape_mnk = (
            self.tile_m,
            self.tile_n,
            cute.size(a_smem_layout_staged.outer.shape, mode=[1]),
        )
        tile_coord = (m_block, n_block, None)

        gA_mk = cute.local_tile(mA_mk, tile_shape_mnk, tile_coord, proj=(1, None, 1))
        gB_nk = cute.local_tile(mB_nk, tile_shape_mnk, tile_coord, proj=(None, 1, 1))
        gC_mn = cute.local_tile(mC_mn, tile_shape_mnk, tile_coord, proj=(1, 1, None))

        # TMA partition A
        sA_for_tma = cute.group_modes(sA, 0, 2)
        gA_for_tma = cute.group_modes(gA_mk, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sA_for_tma,
            gA_for_tma,
        )

        # TMA partition B
        sB_for_tma = cute.group_modes(sB, 0, 2)
        gB_for_tma = cute.group_modes(gB_nk, 0, 2)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            sB_for_tma,
            gB_for_tma,
        )

        # MMA partitions
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        warp_group_thread_layout = cute.make_layout(1, stride=128)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        # Accumulator
        acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        acc = cute.make_fragment(acc_shape, cutlass.Float32)

        # GEMM mainloop: TMA + WGMMA
        k_tile_cnt = cute.size(gA_mk, mode=[2])
        prefetch_cnt = cutlass.min(ab_stage, k_tile_cnt)

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, ab_stage
        )

        # Prefetch phase
        if warp_idx == 0:
            for _ in cutlass.range(prefetch_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(producer_state)
                tAgA_k = tAgA[(None, producer_state.count)]
                tAsA_pipe = tAsA[(None, producer_state.index)]
                tBgB_k = tBgB[(None, producer_state.count)]
                tBsB_pipe = tBsB[(None, producer_state.index)]
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
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
        num_k_blocks = cute.size(tCrA, mode=[2])

        for _k_tile in cutlass.range(k_tile_cnt, unroll=1):
            peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)
            mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, consumer_read_state.index)
                tCrA_phase = tCrA[k_block_coord]
                tCrB_phase = tCrB[k_block_coord]
                cute.gemm(tiled_mma, acc, tCrA_phase, tCrB_phase, acc)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            mainloop_pipeline.consumer_release(consumer_release_state)
            consumer_read_state.advance()
            consumer_release_state.advance()

            # Continue producer
            if warp_idx == 0 and producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(producer_state)
                tAgA_k = tAgA[(None, producer_state.count)]
                tAsA_pipe = tAsA[(None, producer_state.index)]
                tBgB_k = tBgB[(None, producer_state.count)]
                tBsB_pipe = tBsB[(None, producer_state.index)]
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # Epilogue: store GEMM output via Reg -> SMEM -> GMEM (TMA S2G)
        cute.arch.sync_threads()

        # Reuse A's SMEM for epilogue output staging
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self._dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR,
            elem_ty_d=self._dtype,
            elem_ty_acc=cutlass.Float32,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                utils.LayoutEnum.ROW_MAJOR.is_m_major_c(),
                4,
            ),
            self._dtype,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_atom,
        )

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(acc)

        # Allocate D registers
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, cutlass.Float32)
        size_tRS_rD = cute.size(tRS_rD)

        # TMA store partitions
        sC_for_tma = cute.group_modes(sC, 0, 2)
        tCgC_for_tma = cute.zipped_divide(gC_mn, epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma,
            tCgC_for_tma,
        )

        epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
        epi_tile_shape = tCgC_for_tma.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        # Initialize TMA store pipeline
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_threads
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=cute.size(tRS_sD, mode=[3]),
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from accumulator to D registers
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            # Type conversion FP32 -> output dtype
            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self._dtype)
            d_vec = tRS_rD.load()
            tRS_rD_out.store(d_vec.to(self._dtype))

            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            pipeline.sync(barrier_id=1)

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # TMA store: SMEM -> GMEM
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()

            pipeline.sync(barrier_id=1)

        if warp_idx == 0:
            c_pipeline.producer_tail()
