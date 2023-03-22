// @lint-ignore-every LICENSELINT
// clang-format off

/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "classic_b2b_bmm/thread/linear_combination_triu.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Modified version of MmaTensorOpFragmentIterator that can zero out upper triangular
// portion of output matrix.
template <typename MmaTensorOpFragmentIterator_, int ThreadBlockShapeM_>
class TriuMmaTensorOpFragmentIterator {
 public:

  /// Shape of warp tile to load (concept: MatrixShape)
  using Shape = typename MmaTensorOpFragmentIterator_::Shape;

  /// Shape of the warp accumulation tile (concept: MatrixShape)
  using AccumulatorShape = typename MmaTensorOpFragmentIterator_::AccumulatorShape;

  /// KBlocks columns to compute residual
  static int const kKBlockColumn = MmaTensorOpFragmentIterator_::kKBlockColumn;

  /// Accumulator Element type
  using ElementAccumulator = typename MmaTensorOpFragmentIterator_::ElementAccumulator;

  /// Element type
  using Element = typename MmaTensorOpFragmentIterator_::Element;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = typename MmaTensorOpFragmentIterator_::InstructionShape;

  /// Output operation on fragment
  using OutputOp = thread::LinearCombinationTriu<
    typename MmaTensorOpFragmentIterator_::OutputOp::ElementOutput,
    MmaTensorOpFragmentIterator_::OutputOp::kCount,
    ThreadBlockShapeM_,
    typename MmaTensorOpFragmentIterator_::OutputOp::ElementOutput,
    typename MmaTensorOpFragmentIterator_::OutputOp::ElementCompute
  >;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");
    static_assert(
        AccumulatorShape::kRow == Shape::kRow,
        "Rows of Warp Accumulator must be the same as rows of warp");
    static_assert(
        !(AccumulatorShape::kColumn % Shape::kColumn),
        "Shape of Warp Accumulator must be divisible by warp shape.");
    static_assert(
        !(kKBlockColumn % Shape::kColumn),
        "KBlock size must be divisible by warp shape.");

    /// Number of times this iterator can be incremented
    static int const kIterations = AccumulatorShape::kCount / Shape::kCount;
  };

private:

  static int const kElementsPerAccess = InstructionShape::kM * InstructionShape::kN / kThreads;

  /// Number of mma operations performed by a warp
  using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                    Shape::kColumn / InstructionShape::kN>;
  /// Number of mma operations performed by the entire accumulator
  using AccumulatorIterations = MatrixShape<AccumulatorShape::kRow / InstructionShape::kM,
                                              AccumulatorShape::kColumn / InstructionShape::kN>;

  /// Number of K iterations
  static int const kKBlockIterations = (AccumulatorShape::kColumn + kKBlockColumn - 1) / kKBlockColumn;
  static int const kResidualColumn = AccumulatorShape::kColumn - (kKBlockIterations - 1) * kKBlockColumn;
  static int const kKBlockColumnIterations = kKBlockColumn / Shape::kColumn
                                     * (AccumulatorShape::kRow / Shape::kRow);
  static int const kResidualIndex = kResidualColumn / Shape::kColumn
                                     * (AccumulatorShape::kRow / Shape::kRow);

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<Element, Shape::kCount / kThreads>;

  /// Accumulator Fragment object
  using AccumulatorFragment = Array<ElementAccumulator, AccumulatorShape::kCount / kThreads>;

  /// Scale Bias Element Type
  using ElementScaleBias = typename OutputOp::ElementCompute;

  /// Scale Bias Fragment object
  using ScaleBiasFragment = Array<ElementScaleBias, InstructionShape::kM * InstructionShape::kK / kThreads>;


private:

  /// Internal access type
  using AccessType = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentAccessType = Array<Element, kElementsPerAccess>;

  using ScaleBiasAccessType = Array<ElementScaleBias, kElementsPerAccess>;

private:
  //
  // Data members
  //

  /// Internal index
  int index_;

  /// Used to access residual tile first
  bool is_residual_tile_;

  OutputOp output_op;

public:
  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  TriuMmaTensorOpFragmentIterator()
      : index_(0), is_residual_tile_(true), output_op() {}

  /// Add offset
  CUTLASS_HOST_DEVICE
  void add_offset(int index_offset) {
    index_ += index_offset;
    if(is_residual_tile_ && index_ >= kKBlockColumnIterations) {
      index_ = index_ - kKBlockColumnIterations + kResidualIndex;
      is_residual_tile_ = false;
    }
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  TriuMmaTensorOpFragmentIterator &operator++() {
    add_offset(1);
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  TriuMmaTensorOpFragmentIterator &operator--() {
    add_offset(-1);
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    if (output_op.is_source_needed()) //beta must be zero
      assert(0);

    FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaIterations::kColumn; n++) {
      for (int m = 0; m < MmaIterations::kRow; m++) {
        if(!(is_residual_tile_ && index_ >= kResidualIndex)) {
            frag_ptr[m * MmaIterations::kColumn + n] = output_op(
              frag_ptr[m * MmaIterations::kColumn + n],
              index_,
              n,
              m
            );
        }
      }
    }
  }

};

}
}
}
