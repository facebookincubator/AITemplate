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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
    This file is adapted from https://github.com/NVIDIA/cutlass/tree/master/examples/13_two_tensor_op_fusion.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

#include "classic_b2b_bmm/kernel/b2b_batched_gemm.h"
#include "classic_b2b_bmm/kernel/default_b2b_batched_gemm.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Layout type for B1 matrix operand
    typename LayoutB1_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape0_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp0_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Epilogue output operator
    typename EpilogueOutputOp1_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmBatchedIdentityThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Apply upper triangular causal mask after first gemm
    bool CausalMaskAfterGemm0 = false,
    /// Stage accumulator in shared memory
    bool SmemAccumulator = false,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator>
class B2bGemmBatched {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using LayoutB1 = LayoutB1_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape0 = ThreadblockShape0_;
  using ThreadblockShape1 = ThreadblockShape1_;
  using WarpShape0 = WarpShape0_;
  using WarpShape1 = WarpShape1_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp0 = EpilogueOutputOp0_;
  using EpilogueOutputOp1 = EpilogueOutputOp1_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp1::kCount;
  static bool const kCausalMaskAfterGemm0 = CausalMaskAfterGemm0;

  /// Derived types
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;
  using LayoutScaleBias = layout::RowMajor;

  /// Define the kernel
  using B2bGemmBatchedKernel = typename kernel::DefaultB2bGemmBatched<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    LayoutB1,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    ThreadblockSwizzle,
    kStages,
    Operator,
    CausalMaskAfterGemm0,
    SmemAccumulator
  >::B2bGemmBatchedKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size_0;
    GemmCoord problem_size_1;
    TensorRef<ElementA const, LayoutA> ref_A0;
    int64_t stride_A0;
    TensorRef<ElementB const, LayoutB> ref_B0;
    int64_t stride_B0;
    TensorRef<ElementC const, LayoutC> ref_C0;
    int64_t stride_C0;
    TensorRef<ElementB const, LayoutB1> ref_B1;
    int64_t stride_B1;
    TensorRef<ElementC const, LayoutC> ref_C1;
    int64_t stride_C1;
    TensorRef<ElementC, LayoutC> ref_D1;
    int64_t stride_D1;
    int batch_count;
    typename EpilogueOutputOp0::Params epilogue0;
    typename EpilogueOutputOp1::Params epilogue1;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() {

    }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_0_,
      GemmCoord problem_size_1_,
      TensorRef<ElementA const, LayoutA> ref_A0_,
      int64_t stride_A0_,
      TensorRef<ElementB const, LayoutB> ref_B0_,
      int64_t stride_B0_,
      TensorRef<ElementC const, LayoutC> ref_C0_,
      int64_t stride_C0_,
      TensorRef<ElementB const, LayoutB1> ref_B1_,
      int64_t stride_B1_,
      TensorRef<ElementC const, LayoutC> ref_C1_,
      int64_t stride_C1_,
      TensorRef<ElementC, LayoutC> ref_D1_,
      int64_t stride_D1_,
      int batch_count_,
      typename EpilogueOutputOp0::Params epilogue0_ =
        typename EpilogueOutputOp0::Params(),
      typename EpilogueOutputOp1::Params epilogue1_ =
        typename EpilogueOutputOp1::Params()
    ):
      problem_size_0(problem_size_0_),
      problem_size_1(problem_size_1_),
      ref_A0(ref_A0_),
      stride_A0(stride_A0_),
      ref_B0(ref_B0_),
      stride_B0(stride_B0_),
      ref_C0(ref_C0_),
      stride_C0(stride_C0_),
      ref_B1(ref_B1_),
      stride_B1(stride_B1_),
      ref_C1(ref_C1_),
      stride_C1(stride_C1_),
      ref_D1(ref_D1_),
      stride_D1(stride_D1_),
      batch_count(batch_count_),
      epilogue0(epilogue0_),
      epilogue1(epilogue1_) {

    }
  };

private:

  /// Kernel parameters object
  typename B2bGemmBatchedKernel::Params params_;

public:

  /// Constructs the GEMM.
  B2bGemmBatched() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    Status status = B2bGemmBatchedKernel::can_implement(
      args.problem_size_0,
      args.problem_size_1,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    return 0;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size_0,
      {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
      args.batch_count);

    // Initialize the Params structure
    params_ = typename B2bGemmBatchedKernel::Params{
      args.problem_size_0,
      args.problem_size_1,
      grid_shape,
      args.ref_A0.non_const_ref(),
      args.stride_A0,
      args.ref_B0.non_const_ref(),
      args.stride_B0,
      args.ref_C0.non_const_ref(),
      args.stride_C0,
      args.ref_B1.non_const_ref(),
      args.stride_B1,
      args.ref_C1.non_const_ref(),
      args.stride_C1,
      args.ref_D1,
      args.stride_D1,
      args.batch_count,
      args.epilogue0,
      args.epilogue1
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    params_.ref_A0.reset(args.ref_A0.non_const_ref().data());
    params_.ref_B0.reset(args.ref_B0.non_const_ref().data());
    params_.ref_C0.reset(args.ref_C0.non_const_ref().data());
    params_.ref_B1.reset(args.ref_B1.non_const_ref().data());
    params_.ref_C1.reset(args.ref_C1.non_const_ref().data());
    params_.ref_D1.reset(args.ref_D1.data());
    params_.output_op_0 = args.epilogue0;
    params_.output_op_1 = args.epilogue1;

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(B2bGemmBatchedKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename B2bGemmBatchedKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<B2bGemmBatchedKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<B2bGemmBatchedKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
