//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

namespace cutlass {

template <
    typename ElementA_,
    typename LayoutA_,
    int kAlignmentA,
    typename ElementB_,
    typename LayoutB_,
    int kAlignmentB,
    typename ElementC_,
    int kAlignmentC,
    typename OperatorClass,
    typename ArchTag,
    typename ElementAccumulator,
    int kStages,
    typename ThreadblockShape_,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueFunctorOp,
    typename ThreadblockSwizzle,
    typename ElementNorm_ = float,
    typename ElementSum_ = float,
    typename ElementSoftmax_ = ElementC_>

class GemmSoftmaxUniversal {
 public:
  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementCompute = ElementAccumulator;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoftmax_;
  using ElementSoftmaxCompute = float;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

  using ThreadblockShape = ThreadblockShape_;

  static int const kAlignment = kAlignmentA;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  /// Linear scaling operator
  // using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
  //   ElementC,
  //   kAlignment,
  //   ElementCompute,
  //   ElementCompute
  // >;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // This is a mandatory data type for the atomic reduction in the GEMM epilogue
  // to function.

  // using ElementN = float;
  using ElementNorm = ElementNorm_;

  using ApplyShape = MatrixShape<1, 1024>;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutN = cutlass::layout::RowMajor;
  using LayoutS = cutlass::layout::RowMajor;
  using LayoutSoft = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;
  using TensorRefN = TensorRef<ElementNorm, LayoutN>;
  using TensorRefSum = TensorRef<ElementSum, LayoutS>;
  using TensorRefSoft = TensorRef<ElementSoft, LayoutSoft>;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // basic GEMM kernel
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementC,
      LayoutC,
      ElementCompute,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueFunctorOp,
      ThreadblockSwizzle,
      kStages,
      true,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OperatorClass,
          ArchTag,
          ElementA,
          ElementB,
          ElementC,
          ElementCompute>::Operator,
      cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor =
      typename cutlass::epilogue::threadblock::EpilogueVisitorSoftmax<
          ThreadblockShape,
          DefaultGemmKernel::kThreadCount,
          typename DefaultGemmKernel::Epilogue::OutputTileIterator,
          ElementCompute,
          ElementNorm,
          ElementSum,
          ElementSoftmaxCompute,
          EpilogueFunctorOp>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<
          EpilogueVisitor,
          typename DefaultGemmKernel::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel = gemm::kernel::GemmWithEpilogueVisitor<
      typename DefaultGemmKernel::Mma,
      Epilogue,
      ThreadblockSwizzle>;

  // Softmax kernel
  using SoftmaxApplyKernel = kernel::ApplySoftmax<
      ElementC,
      ElementNorm,
      ElementSum,
      ElementSoft,
      ElementSoftmaxCompute,
      kAlignmentC,
      ApplyShape>;

  using ApplyFinalReductionKernel =
      cutlass::reduction::kernel::ApplySoftmaxFinalReduction<
          ElementNorm,
          ElementSum,
          ElementSoftmaxCompute,
          ThreadblockShape>;

 public:
  /// Arguments class
  struct Arguments {
    typename GemmKernel::Arguments gemm;

    typename SoftmaxApplyKernel::Arguments softmax;
    typename ApplyFinalReductionKernel::Arguments reduction;
    cutlass::gemm::GemmCoord extent;

    //
    // Methods
    //
    Arguments() {}

    Arguments(
        cutlass::gemm::GemmCoord problem_size,
        int32_t batch_count_,
        TensorRefA ref_A_,
        TensorRefB ref_B_,
        TensorRefC ref_C_,
        TensorRefC ref_D_,
        typename EpilogueFunctorOp::Params linear_scaling,
        TensorRefN ref_N_,
        TensorRefSum ref_S_,
        TensorRefSoft ref_Softmax_,
        int64_t batch_stride_A_ = 0,
        int64_t batch_stride_B_ = 0,
        int64_t batch_stride_C_ = 0,
        int64_t batch_stride_D_ = 0,
        int64_t batch_stride_Max_ = 0,
        int64_t batch_stride_Sum_ = 0,
        int64_t batch_stride_Softmax_ = 0)
        : gemm(
              cutlass::gemm::GemmUniversalMode::kBatched,
              problem_size,
              batch_count_,
              ref_A_,
              ref_B_,
              ref_C_,
              ref_D_,
              ref_N_.data(),
              ref_S_.data(),
              batch_stride_A_,
              batch_stride_B_,
              typename EpilogueVisitor::Arguments(
                  linear_scaling,
                  batch_stride_C_,
                  batch_stride_D_,
                  batch_stride_Max_,
                  batch_stride_Sum_)),
          reduction(
              problem_size,
              ref_N_.data(),
              ref_S_.data(),
              batch_stride_Max_,
              batch_stride_Sum_),
          softmax(
              MatrixCoord(problem_size.m(), problem_size.n()),
              batch_count_,
              ref_D_,
              ref_N_,
              ref_S_,
              ref_Softmax_,
              batch_stride_D_,
              batch_stride_Max_,
              batch_stride_Sum_,
              batch_stride_Softmax_),
          extent(problem_size) {}
  };

  struct Params {
    typename GemmKernel::Params gemm;

    typename SoftmaxApplyKernel::Params softmax;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord extent;

    //
    // Methods
    //
    Params() {}

    Params(Arguments const& args)
        : gemm(args.gemm),
          reduction(args.reduction),
          softmax(args.softmax),
          extent(MatrixCoord(args.extent.m(), args.extent.n())) {}
  };

 public:
  // Gemm

  //
  // Methods
  //

 private:
  Params params_;

 public:
  /// Ctor
  GemmSoftmaxUniversal() {}

  /// Initialize
  Status initialize(Arguments const& args) {
    params_ = Params(args);

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {
    //
    // Launch the GEMM + max kernel
    //

    dim3 gemm_grid =
        ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);

    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cudaError_t result;

    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(
          cutlass::Kernel<GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel>
        <<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the ApplyFinalReductionKernel
    //

    int thread_per_block = 128;
    int block_per_row =
        (params_.extent.row() + thread_per_block - 1) / thread_per_block;
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row =
          (params_.extent.row() + thread_per_block - 1) / thread_per_block;
    }

    dim3 final_reduction_grid(
        block_per_row, 1, params_.softmax.args.batch_count);
    dim3 final_reduction_block(thread_per_block);

    Kernel<ApplyFinalReductionKernel>
        <<<final_reduction_grid,
           final_reduction_block,
           sizeof(typename ApplyFinalReductionKernel::SharedStorage),
           stream>>>(params_.reduction);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the SoftmaxApplyKernel
    //

    dim3 apply_block(
        SoftmaxApplyKernel::ApplyShape::kColumn,
        SoftmaxApplyKernel::ApplyShape::kRow);

    int threadblock_rows = SoftmaxApplyKernel::ApplyShape::kRow;
    int threadblock_columns = SoftmaxApplyKernel::ApplyShape::kColumn *
        SoftmaxApplyKernel::kAlignment;

    dim3 apply_grid(
        (params_.softmax.args.extent.row() + threadblock_rows - 1) /
            threadblock_rows,
        (params_.softmax.args.extent.column() + threadblock_columns - 1) /
            threadblock_columns,
        params_.softmax.args.batch_count);

    Kernel<SoftmaxApplyKernel>
        <<<apply_grid,
           apply_block,
           sizeof(typename SoftmaxApplyKernel::SharedStorage),
           stream>>>(params_.softmax);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

} // namespace cutlass
