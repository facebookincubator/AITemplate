/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holdvr nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
 * Mostly copied from
 * http://github.com/NVIDIA/cutlass/tree/master/examples/41_fused_multi_head_attention
 */

#pragma once

#include "cutlass/functional.h"
#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"
#include "cutlass/matrix_shape.h"
#include "fmha_style_b2b_bmm/gemm_kernel_utils.h"

/* Iterates on the accumulator and corresponding position on result matrix

All of this is done on registers, before we store all of this
on shared memory for the next matmul with Value.

We have multiple implementations, because each configuration has a different way
of iterating in the accumulators.
*/

template <typename BASE, typename T, typename accum_t, int kWarpSize>
struct RegisterOps {};

template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterSm80
    : RegisterOps<
          AttentionScalingCoefsUpdaterSm80<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  static int const kElementsPerAccess = InstructionShape::kN / 4;
  static int const kRowsPerTile = 8;
  static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    return cutlass::MatrixCoord(
        quad + tile_offset.row() * Shape::kRow,
        lane_in_quad * kElementsPerAccess +
            tile_offset.column() * Shape::kColumn);
  }

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    // See cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < kAccumulatorRows; ++row) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
            row * kRowsPerTile + lane_offset.row();
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
              (mma_n * Policy::MmaIterations::kRow + mma_m);
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn +
                col + lane_offset.column();
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            op(accum_m, accum_n, idx);
          }
        }

        endRow(accum_m);
      }
    }
  }

  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    // In each warp, 4 threads will work on the same row
    // - the ones with the same `quad`
    auto otherV = __shfl_xor_sync(0xffffffff, myValue, 1);
    myValue = fn(myValue, otherV);
    otherV = __shfl_xor_sync(0xffffffff, myValue, 2);
    myValue = fn(myValue, otherV);
    int lane_in_quad = (lane_id & 3);
    return lane_in_quad == 0;
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterVolta
    : RegisterOps<
          AttentionScalingCoefsUpdaterVolta<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  using Element = accum_t;

  static int const kElementsPerPartial = 4;
  using EleShapePerPatial = typename cutlass::platform::conditional<
      cutlass::platform::is_same<Element, float>::value,
      cutlass::MatrixShape<2, 2>,
      cutlass::MatrixShape<1, 4>>::type;
  static int const kElementsPerMma = 8;
  static int const kAccumulatorPatials = 2;
  using QuadShapePerPatialMma = cutlass::MatrixShape<4, 4>;

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    int accum_m, accum_n;

    if (cutlass::platform::is_same<Element, float>::value) {
      // (quad[2],quad[0])+lane_in_quad[0]
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + (lane_in_quad & 1);
      // (quad[1])+lane_in_quad[1]
      accum_n =
          ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials +
          (lane_in_quad & 2);
    } else {
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 +
          lane_in_quad; // (quad[2],quad[0])
      accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials;
    }
    return cutlass::MatrixCoord(
        accum_m + tile_offset.row() * Shape::kRow,
        accum_n + tile_offset.column() * Shape::kColumn);
  }

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    CUTLASS_PRAGMA_UNROLL
    for (int tile_m = 0; tile_m < Policy::TileIterations::kRow; ++tile_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
          int accum_m = tile_m * Policy::InterleavedTile::kRow +
              mma_m * QuadShapePerPatialMma::kRow + m * 2 + lane_offset.row();
          beginRow(accum_m);

          CUTLASS_PRAGMA_UNROLL
          for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn;
               ++tile_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn;
                 ++mma_n) {
              CUTLASS_PRAGMA_UNROLL
              for (int p = 0; p < kAccumulatorPatials; ++p) {
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < EleShapePerPatial::kColumn; ++n) {
                  int mma_accum_start =
                      (((tile_n * Policy::TileIterations::kRow + tile_m) *
                            Policy::MmaIterations::kColumn +
                        mma_n) *
                           Policy::MmaIterations::kRow +
                       mma_m) *
                      kElementsPerMma;
                  int accum_n = tile_n * Policy::InterleavedTile::kColumn +
                      mma_n * QuadShapePerPatialMma::kColumn +
                      p * Policy::InterleavedTile::kColumn / 2 + n +
                      lane_offset.column();
                  int idx = mma_accum_start + p * kElementsPerPartial +
                      m * EleShapePerPatial::kColumn + n;
                  op(accum_m, accum_n, idx);
                }
              }
            }
          }
          endRow(accum_m);
        }
      }
    }
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterSimt
    : RegisterOps<
          AttentionScalingCoefsUpdaterSimt<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  using Policy = typename T::Policy;
  using Iterations = typename T::Iterations;
  using Element = typename T::Element;
  using Delta = typename T::Delta;
  using Shape = typename T::Shape;
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
        int accum_m = mma_m * Delta::kRow + m + lane_offset.row();
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
          int accum_n =
              mma_n * Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN +
              lane_offset.column();
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
            int idx = n +
                Policy::LaneMmaShape::kN *
                    (mma_n +
                     Iterations::kColumn *
                         (m + mma_m * Policy::LaneMmaShape::kM));
            op(accum_m, accum_n + n, idx);
          }
        }
        endRow(accum_m);
      }
    }
  }

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    static_assert(
        cutlass::platform::is_same<
            typename Policy::LaneLayout,
            cutlass::layout::RowMajorInterleaved<1>>::value,
        "");
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    cutlass::MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
        cutlass::MatrixCoord(Policy::LaneMmaShape::kM,
                             Policy::LaneMmaShape::kN);
    return lane_offset +
        tile_offset * cutlass::MatrixCoord(Shape::kRow, Shape::kColumn);
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater;

// Simt
template <typename S, typename P, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaSimtTileIterator<
        S,
        cutlass::gemm::Operand::kC,
        accum_t,
        cutlass::layout::RowMajor,
        P,
        1,
        1>,
    accum_t,
    kWarpSize> {
  using Iterator = typename cutlass::gemm::warp::MmaSimtTileIterator<
      S,
      cutlass::gemm::Operand::kC,
      accum_t,
      cutlass::layout::RowMajor,
      P,
      1,
      1>;
  using Updater =
      AttentionScalingCoefsUpdaterSimt<Iterator, accum_t, kWarpSize>;
};

// TensorOp - Volta
template <typename S1, typename S2, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
        S1,
        accum_t,
        cutlass::layout::RowMajor,
        S2,
        cutlass::MatrixShape<1, 1>>,
    accum_t,
    kWarpSize> {
  using Iterator =
      typename cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
          S1,
          accum_t,
          cutlass::layout::RowMajor,
          S2,
          cutlass::MatrixShape<1, 1>>;
  using Updater =
      AttentionScalingCoefsUpdaterVolta<Iterator, accum_t, kWarpSize>;
};

// TensorOp - Sm75+
template <
    typename S1,
    typename S2,
    typename S3,
    typename accum_t,
    int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
        S1,
        accum_t,
        cutlass::layout::RowMajor,
        S2,
        S3>,
    accum_t,
    kWarpSize> {
  using Iterator =
      typename cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
          S1,
          accum_t,
          cutlass::layout::RowMajor,
          S2,
          S3>;
  using Updater =
      AttentionScalingCoefsUpdaterSm80<Iterator, accum_t, kWarpSize>;
};
