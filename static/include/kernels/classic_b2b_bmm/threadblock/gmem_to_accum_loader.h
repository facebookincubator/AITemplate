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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

  The shared memory resource is time-sliced across warps.

  NOTE: Copied from cutlass/epilogue/threadblock/epilogue.h and modified to essentially
  inverse the direction of the epilogue. See https://github.com/NVIDIA/cutlass/issues/784
  for details.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/util/index_sequence.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {


////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)
>
class GmemToAccumLoader :
  public EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>,
  public EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>
{

public:

  using Base = EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>;

  using BaseStreamK = EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// Number of warps per block
  using WarpCount = typename Base::WarpCount;

  /// Number of threads per block
  static int const kBlockThreads = 32 * WarpCount::kCount;

  /// Per-thread accumulator tile type
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Numerical accumulation element type
  using ElementAccumulator = typename WarpMmaOperator::ElementC;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Vector type used by the global output iterator
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Vector type used by the shared output iterator
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

public:


  static_assert(SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess),
    "Divisibility");

  static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");

  static_assert(kPartitionsK == 1, "Must be exactly 1.");

private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Thread index in the threadblock
  int thread_idx;

  /// Warp index in the threadblock
  int warp_idx;

public:

  /// Constructor
  CUTLASS_DEVICE
  GmemToAccumLoader(
      typename Base::SharedStorage &shared_storage,   ///< Shared storage object
      int thread_idx,                                 ///< ID of a thread within the threadblock
      int warp_idx,                                   ///< ID of warp within threadblock
      int lane_idx)                                   ///< Id of thread within warp
  :
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      BaseStreamK(thread_idx),
      shared_load_iterator_(shared_storage.reference(), thread_idx),
      thread_idx(thread_idx),
      warp_idx(warp_idx)
  {}

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                ///< Output operator
    AccumulatorTile &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator )      ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
  {
    if (!output_op.is_source_needed())
    {
      source_iterator.clear_mask();
      __syncthreads();  // Dummy (CUDA 11.0)
    }

    // Source-fragment data (zero-initialized for scenarios where the
    // output operator allows us to skip loading it from global input)
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; iter += Base::kFragmentsPerIteration)
    {

      //
      // Load fragments from shared memory
      //

      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p)
      {
        // Load addend source fragment from global memory to aligned register fragment.
        source_iterator.load(source_fragment);
        ++source_iterator;

        // Store data in register fragment to shared memory.
        shared_load_iterator_.store(source_fragment);

        if (p < Base::kFragmentsPerIteration - 1)
        {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }

      //
      // Convert and store fragment
      //

      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p)
      {

        typename AccumulatorFragmentIterator::Fragment accum_fragment;
        typename AccumulatorFragmentIterator::Fragment source_accum_fragment;
        typename AccumulatorFragmentIterator::Fragment output_accum_fragment;

        // Load from shared memory to "unaligned" accumulator fragment.
        this->warp_tile_iterator_.load(source_accum_fragment);

        // Load from accumulators to accumulator fragment.
        accum_fragment_iterator.load(accum_fragment);

        // Store result of computation to accumulators.
        apply_output_operator(output_accum_fragment, output_op, accum_fragment, source_accum_fragment);
        accum_fragment_iterator.store(output_accum_fragment);

        ++accum_fragment_iterator;

        if (p < Base::kFragmentsPerIteration - 1) {
          this->warp_tile_iterator_.add_pointer_offset(kSmemPointerOffset);
        }

      }

      if (Base::kFragmentsPerIteration > 1) {
        this->warp_tile_iterator_.add_pointer_offset(kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }

    }

  }

private:

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator(
    typename AccumulatorFragmentIterator::Fragment &output_fragment,
    OutputOp const &output_op,                    ///< Output operator
    typename AccumulatorFragmentIterator::Fragment const &accum_fragment,
    typename AccumulatorFragmentIterator::Fragment const &source_fragment)
  {

    OutputAccessType *output_frag_ptr =
      reinterpret_cast<OutputAccessType *>(&output_fragment);

    AccumulatorAccessType const *compute_frag_ptr =
      reinterpret_cast<AccumulatorAccessType const *>(&accum_fragment);

    OutputAccessType const *source_frag_ptr =
      reinterpret_cast<OutputAccessType const *>(&source_fragment);

    int const kOutputOpIterations =
      AccumulatorFragmentIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i)
    {
      // Call the output operator
      output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
    }
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
