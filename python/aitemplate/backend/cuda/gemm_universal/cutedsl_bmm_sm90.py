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
CuTeDSL SM90 (Hopper) implementation of Batched Matrix Multiply (BMM).

Supports configurable layouts:
  RRR: A[B, M, K] row-major, B[B, K, N] row-major, C[B, M, N] row-major
  CCR: A[B, K, M] col-major, B[B, N, K] col-major, C[B, M, N] row-major
  (and other combinations)

Optional residual add for _add variants:
  C[B, M, N] = A @ B + D[B, M, N]

Uses Hopper-specific features:
  - TMA for efficient GMEM <-> SMEM transfers
  - WGMMA instructions
  - PipelineTmaAsync for multi-stage pipeline
  - TMA store for epilogue
"""

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils


class BmmSm90Kernel:
    """CuTeDSL SM90 BMM kernel using TMA + WGMMA.

    C[B, M, N] = A @ B  (+ D if has_d)

    Layout controlled by a_row_major/b_row_major:
      RRR: a_row_major=True,  b_row_major=True  -> A[B,M,K], B[B,K,N]
      CCR: a_row_major=False, b_row_major=False  -> A[B,K,M], B[B,N,K]

    Grid: (ceil(M/tile_m), ceil(N/tile_n), B)

    Parameters
    ----------
    tile_m : int
        Threadblock tile in M.
    tile_n : int
        Threadblock tile in N.
    a_row_major : bool
        True if A is row-major (M,K), False if col-major (K,M).
    b_row_major : bool
        True if B is row-major (K,N), False if col-major (N,K).
    has_d : bool
        If True, adds residual tensor D to the output.
    """

    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
        a_row_major: bool = True,
        b_row_major: bool = True,
        has_d: bool = False,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.a_row_major = a_row_major
        self.b_row_major = b_row_major
        self.has_d = has_d

        self.tile_k = 64
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
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mD: cute.Tensor,  # residual (ignored if has_d=False)
        B_dim: cutlass.Int32,
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._dtype: Type[cutlass.Numeric] = mA.element_type

        a_layout = (
            utils.LayoutEnum.ROW_MAJOR
            if self.a_row_major
            else utils.LayoutEnum.COL_MAJOR
        )
        b_layout = (
            utils.LayoutEnum.ROW_MAJOR
            if self.b_row_major
            else utils.LayoutEnum.COL_MAJOR
        )
        c_layout = utils.LayoutEnum.ROW_MAJOR

        # SM90 tiled MMA — M dimension of atom is always 64
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

        # SMEM layouts
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, self._dtype, tile_k),
            self._dtype,
        )
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, self._dtype, tile_k),
            self._dtype,
        )

        # Pipeline stages
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

        # Epilogue SMEM
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

        # TMA atoms — created on the full 3D tensors
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

        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(mC.shape),
            epi_tile,
        )
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

        # Grid: (M_blocks, N_blocks, Batch)
        m_blocks = cute.ceil_div(M, self.tile_m)
        n_blocks = cute.ceil_div(N, self.tile_n)
        grid_dim = (m_blocks, n_blocks, B_dim)

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            mC,  # raw C for post-epilogue D-add
            mD,  # residual tensor (ignored if has_d=False)
            B_dim,
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
        mC_raw: cute.Tensor,
        mD_raw: cute.Tensor,
        B_dim: cutlass.Int32,
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
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        m_block, n_block, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ----- Shared memory setup -----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # ----- Pipeline setup -----
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        num_warps = self.num_threads // 32
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_warps
        )

        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        pl = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        cute.arch.sync_threads()

        # ----- Tile shape -----
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        tile_k = mma_inst_shape_k * 4
        k_tiles = cute.ceil_div(K, tile_k)

        # ----- Tile the 3D tensors for this CTA -----
        # For A: project out the batch dim + tile M and K
        # For B: project out the batch dim + tile N and K
        # tile_coord includes batch_idx
        if self.a_row_major:
            # A[B, M, K]: tile_shape=(1, tile_m, tile_k), coord=(batch_idx, m_block, _)
            tile_coord_a = (batch_idx, m_block, cute._)
            gA_mk = cute.local_tile(
                mA_mk, (1, self.tile_m, tile_k), tile_coord_a, proj=(None, 1, 1)
            )
        else:
            # A[B, K, M]: tile_shape=(1, tile_k, tile_m), coord=(batch_idx, _, m_block)
            tile_coord_a = (batch_idx, cute._, m_block)
            gA_mk = cute.local_tile(
                mA_mk, (1, tile_k, self.tile_m), tile_coord_a, proj=(None, 1, 1)
            )

        if self.b_row_major:
            # B[B, K, N]: tile_shape=(1, tile_k, tile_n), coord=(batch_idx, _, n_block)
            tile_coord_b = (batch_idx, cute._, n_block)
            gB_nk = cute.local_tile(
                mB_nk, (1, tile_k, self.tile_n), tile_coord_b, proj=(None, 1, 1)
            )
        else:
            # B[B, N, K]: tile_shape=(1, tile_n, tile_k), coord=(batch_idx, n_block, _)
            tile_coord_b = (batch_idx, n_block, cute._)
            gB_nk = cute.local_tile(
                mB_nk, (1, self.tile_n, tile_k), tile_coord_b, proj=(None, 1, 1)
            )

        # C output tile: C[B, M, N]
        tile_coord_c = (batch_idx, m_block, n_block)
        gC_mn = cute.local_tile(
            mC_mn, (1, self.tile_m, self.tile_n), tile_coord_c, proj=(None, 1, 1)
        )

        # ----- TMA partitions -----
        tCgA = tma_atom_a.get_slice(cute._).partition_S(gA_mk)
        tCsA = tma_atom_a.get_slice(cute._).partition_D(sA)
        tCgB = tma_atom_b.get_slice(cute._).partition_S(gB_nk)
        tCsB = tma_atom_b.get_slice(cute._).partition_D(sB)

        # ----- MMA partition -----
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA_mma = thr_mma.partition_A(sA[(None, None, 0)])
        tCsB_mma = thr_mma.partition_B(sB[(None, None, 0)])
        acc = thr_mma.make_fragment_C(cutlass.Float32)
        cute.fill(acc, 0.0)

        # ----- Epilogue setup -----
        tCgC_for_tma_partition = cute.zipped_divide(gC_mn, epi_tile)
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape,
            stride=(epi_tile_shape[1], 1),
        )
        epi_tile_num = cute.size(epi_tile_layout)

        # Retile accumulator for R2S copy
        tiled_copy_r2s = sm90_utils.make_tiled_copy_r2s(
            self._dtype,
            tiled_mma,
        )
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        sC_ptr = cute.recast_ptr(
            sA.iterator,
            epi_smem_layout_staged.inner,
            dtype=self._dtype,
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile_S(acc)

        tRS_rD_layout = cute.make_fragment_like(
            tRS_sD[(None, None, None, 0)],
            cutlass.Float32,
        ).layout
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, cutlass.Float32)
        size_tRS_rD = cute.size(tRS_rD)

        # TMA store partitions
        bSG_sD = tma_atom_c.get_slice(cute._).partition_S(sC)
        bSG_gD = tCgC_for_tma_partition

        # C pipeline for epilogue stores
        c_pipeline = pipeline.PipelineTmaStore(
            stages=cute.size(tRS_sD, mode=[3]),
        )
        c_pipeline.producer_acquire()

        # =====================================================================
        # Mainloop: TMA G2S + WGMMA
        # =====================================================================

        # Initialize pipeline states for producer and consumer
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, ab_stage
        )
        consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, ab_stage
        )
        consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, ab_stage
        )

        # Prefetch first stages
        prefetch_cnt = cutlass.min(ab_stage, k_tiles)
        if warp_idx == 0:
            for k_tile in cutlass.range(prefetch_cnt, unroll=1):
                pl.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    tCgA[(None, cute._, k_tile)],
                    tCsA[(None, cute._, producer_state.index)],
                    tma_bar_ptr=pl.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_b,
                    tCgB[(None, cute._, k_tile)],
                    tCsB[(None, cute._, producer_state.index)],
                    tma_bar_ptr=pl.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                pl.producer_commit(producer_state)
                producer_state.advance()

        # Mainloop
        for _k_tile in cutlass.range(k_tiles, unroll_full=True):
            peek_status = pl.consumer_try_wait(consumer_read_state)
            pl.consumer_wait(consumer_read_state, peek_status)
            cute.arch.warpgroup_fence_operand_reads()

            cute.gemm(
                tiled_mma,
                tCsA_mma[(None, None, None, consumer_read_state.index)],
                tCsB_mma[(None, None, None, consumer_read_state.index)],
                acc,
            )
            cute.arch.warpgroup_commit_batch()
            cute.arch.warpgroup_wait(count=0)
            cute.arch.warpgroup_fence_operand_reads()

            pl.consumer_release(consumer_release_state)
            consumer_read_state.advance()
            consumer_release_state.advance()

            # Continue producer for remaining tiles
            if warp_idx == 0 and producer_state.count < k_tiles:
                pl.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    tCgA[(None, cute._, producer_state.count)],
                    tCsA[(None, cute._, producer_state.index)],
                    tma_bar_ptr=pl.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_b,
                    tCgB[(None, cute._, producer_state.count)],
                    tCsB[(None, cute._, producer_state.index)],
                    tma_bar_ptr=pl.producer_get_barrier(producer_state),
                    mcast_mask=0,
                )
                pl.producer_commit(producer_state)
                producer_state.advance()

        # =====================================================================
        # Epilogue: Reg -> SMEM -> GMEM via TMA store
        # =====================================================================
        cute.arch.sync_threads()

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self._dtype)
            d_vec = tRS_rD.load()
            tRS_rD_out.store(d_vec.to(self._dtype))

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

        # =====================================================================
        # Post-epilogue: optional residual add (D tensor)
        # =====================================================================
        if cutlass.const_expr(self.has_d):
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_global,
            )
            cute.arch.sync_threads()

            # Tile the raw C and D tensors for this CTA's batch slice
            gC_raw = cute.local_tile(
                mC_raw,
                (1, self.tile_m, self.tile_n),
                (batch_idx, m_block, n_block),
                proj=(None, 1, 1),
            )
            gD_raw = cute.local_tile(
                mD_raw,
                (1, self.tile_m, self.tile_n),
                (batch_idx, m_block, n_block),
                proj=(None, 1, 1),
            )

            tile_m_size = self.tile_m
            tile_n_size = self.tile_n
            total_elems = tile_m_size * tile_n_size
            num_threads = self.num_threads
            elems_per_thread = cute.ceil_div(total_elems, num_threads)

            for elem_idx in cutlass.range(elems_per_thread, unroll_full=True):
                flat_idx = tidx + elem_idx * num_threads
                if flat_idx < total_elems:
                    row = flat_idx // tile_n_size
                    col = flat_idx % tile_n_size
                    c_val = gC_raw[row, col]
                    d_val = gD_raw[row, col]
                    val = cutlass.Float32(c_val) + cutlass.Float32(d_val)
                    gC_raw[row, col] = self._dtype(val)
