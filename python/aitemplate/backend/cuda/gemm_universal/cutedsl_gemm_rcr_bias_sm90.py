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
CuTeDSL SM90 (Hopper) implementation of GEMM RCR with bias.

Computes: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]

Layout convention (RCR):
  A: Row-major [M, K] (K contiguous) -- K-major
  B: Column-major, stored as [N, K] row-major (K contiguous) -- K-major
  C: Row-major [M, N] (N contiguous)
  Bias: 1D [N], broadcast across all M rows

Uses Hopper-specific features:
  - TMA (Tensor Memory Accelerator) for efficient GMEM <-> SMEM transfers
  - WGMMA (Warpgroup Matrix Multiply-Accumulate) instructions
  - PipelineTmaAsync for multi-stage producer/consumer pipeline
  - TMA store for epilogue output

This is equivalent to torch.nn.functional.linear(A, B, bias=Bias).
"""

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils


class GemmRcrBiasSm90Kernel:
    """CuTeDSL SM90 GEMM RCR with 1D bias kernel and fused epilogue using TMA + WGMMA.

    C[M, N] = epilogue(A[M, K] @ B[N, K]^T + Bias[N])

    Supported fusion_type values (compile-time constant):
      0: identity  -> C = gemm + bias
      1: relu      -> C = relu(gemm + bias)
      2: sigmoid   -> C = sigmoid(gemm + bias)
      3: swish     -> C = silu(gemm + bias) = x * sigmoid(x)

    Tile sizes:
      tile_m=128, tile_n=128, tile_k=64 (MMA_K for SM90)
      1 warp group (128 threads)

    Grid: (ceil(M/tile_m), ceil(N/tile_n), 1)

    The bias and activation are applied after the GEMM mainloop:
    1. GEMM output stored via TMA S2G
    2. Post-epilogue pass: read back from GMEM, add bias + apply activation, write back

    Parameters
    ----------
    tile_m : int
        Threadblock tile size in M dimension.
    tile_n : int
        Threadblock tile size in N dimension.
    fusion_type : int
        Compile-time epilogue activation: 0=none, 1=relu, 2=sigmoid, 3=swish.
    """

    # Epilogue fusion type constants
    FUSION_NONE = 0
    FUSION_RELU = 1
    FUSION_SIGMOID = 2
    FUSION_SWISH = 3

    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
        fusion_type: int = 0,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.fusion_type = fusion_type

        # SM90 tile_k is determined by the MMA instruction
        self.tile_k = 64  # Will be updated in _setup_attributes

        # SM90 uses 1 warp group (128 threads) for standard tile sizes
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
        mBias: cute.Tensor,  # [N] 1D bias
        mC: cute.Tensor,  # [M, N] row-major output
        M: cutlass.Int32,  # M dimension
        N: cutlass.Int32,  # N dimension
        K: cutlass.Int32,  # K dimension
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch."""
        self._dtype: Type[cutlass.Numeric] = mA.element_type

        # Both A and B are row-major (K-major for RCR layout)
        a_layout = utils.LayoutEnum.ROW_MAJOR
        b_layout = utils.LayoutEnum.ROW_MAJOR
        c_layout = utils.LayoutEnum.ROW_MAJOR

        # Setup tiled MMA for SM90
        # SM90 WGMMA instruction always has M=64; for tile_m>64, CuTe's
        # partition automatically creates a repeat mode (tile_m/64) that
        # cute.gemm iterates over.
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
        tile_k = mma_inst_shape_k * 4  # 4 MMA K-blocks per tile
        _ = (self.tile_m, self.tile_n, tile_k)  # tile_shape_mnk for reference

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
        ab_stage = min(ab_stage, 7)  # cap stages

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

        # TMA atoms for A and B (no multicast for cluster (1,1))
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

        # Bias remains 1D -- we will index it in the epilogue using
        # epi-tile N coordinates rather than partition_C on a broadcast view.

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
            mBias,
            mC,
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
        mBias: cute.Tensor,
        mC_raw: cute.Tensor,  # Raw GMEM tensor for post-epilogue bias addition
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
        """SM90 device kernel for GEMM RCR + bias."""
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

        # =====================================================================
        # Partition global tensors for TMA
        # =====================================================================
        tile_shape_mnk = (
            self.tile_m,
            self.tile_n,
            cute.size(a_smem_layout_staged.outer.shape, mode=[1]),
        )
        tile_coord = (m_block, n_block, None)

        # A: (tile_m, tile_k, K_tiles)
        gA_mk = cute.local_tile(mA_mk, tile_shape_mnk, tile_coord, proj=(1, None, 1))
        # B: (tile_n, tile_k, K_tiles)
        gB_nk = cute.local_tile(mB_nk, tile_shape_mnk, tile_coord, proj=(None, 1, 1))
        # C: (tile_m, tile_n)
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

        # =====================================================================
        # GEMM mainloop: TMA + WGMMA
        # =====================================================================
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

        # =====================================================================
        # Epilogue: store GEMM output, then add 1D bias in GMEM
        # =====================================================================

        # The GEMM accumulators are stored to GMEM via the standard SM90
        # Reg -> SMEM (stmatrix) -> GMEM (TMA S2G) pipeline.  After all
        # TMA stores complete, a second pass adds the 1D bias directly in
        # GMEM using simple element-wise loads/stores.
        #
        # NOTE: Applying bias in the accumulator registers before the
        # epilogue would be more efficient but requires mapping each MMA
        # register to its logical (M, N) coordinate.  CuTeDSL's
        # partition_C does not yet produce correct indexing for stride-0
        # (broadcast) tensors on SM90 WGMMA, so we use this post-store
        # approach as a correct fallback.

        # Store output: Reg -> SMEM -> GMEM via TMA store
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

        # =====================================================================
        # Post-epilogue: add 1D bias to C in GMEM
        # =====================================================================
        # Wait for all TMA stores to complete before reading C back
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_global,
        )
        cute.arch.sync_threads()

        # Tile the raw C tensor to get this CTA's output tile
        gC_raw = cute.local_tile(
            mC_raw,
            (self.tile_m, self.tile_n),
            (m_block, n_block),
        )

        # Each thread adds bias to a portion of the CTA's output tile.
        tile_m_size = self.tile_m
        tile_n_size = self.tile_n
        total_elems = tile_m_size * tile_n_size
        num_threads = self.num_threads
        elems_per_thread = cute.ceil_div(total_elems, num_threads)

        n_base = n_block * self.tile_n

        for elem_idx in cutlass.range(elems_per_thread, unroll_full=True):
            flat_idx = tidx + elem_idx * num_threads
            if flat_idx < total_elems:
                row = flat_idx // tile_n_size
                col = flat_idx % tile_n_size
                n_global = n_base + col
                bias_val = mBias[n_global]
                old_val = gC_raw[row, col]
                # Promote to FP32 for accumulation + activation, then
                # convert back to the output dtype before storing.
                val = cutlass.Float32(old_val) + cutlass.Float32(bias_val)
                # Compile-time epilogue activation (no runtime overhead)
                if cutlass.const_expr(self.fusion_type == self.FUSION_RELU):
                    val = cute.arch.fmax(val, cutlass.Float32(0.0))
                elif cutlass.const_expr(self.fusion_type == self.FUSION_SIGMOID):
                    val = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.math.exp(-val)
                    )
                elif cutlass.const_expr(self.fusion_type == self.FUSION_SWISH):
                    val = val / (cutlass.Float32(1.0) + cute.math.exp(-val))
                gC_raw[row, col] = self._dtype(val)
