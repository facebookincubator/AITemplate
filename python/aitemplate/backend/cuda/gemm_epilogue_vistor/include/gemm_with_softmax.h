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
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueFunctorOp,
    typename ThreadblockSwizzle,
    typename ElementSum_ = ElementAccumulator,
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

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

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

  using ElementN = float;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutN = cutlass::layout::RowMajor;
  using LayoutSoft = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;
  using TensorRefN = TensorRef<ElementN, LayoutN>;
  using TensorRefSoft = TensorRef<ElementSoft, LayoutSoft>;

  // using OperatorClass       = cutlass::arch::OpClassTensorOp;
  // using ArchTag             = cutlass::arch::Sm80;
  // static int const kStages  = Stages;
  // using ThreadblockSwizzle =
  // cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // basic GEMM kernel
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementA,
      LayoutA,
      kAlignment,
      ElementB,
      LayoutB,
      kAlignment,
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
  using EpilogueVisitor = kernel::EpilogueVisitorBiasMax<
      ThreadblockShape,
      DefaultGemmKernel::kThreadCount,
      typename DefaultGemmKernel::Epilogue::OutputTileIterator,
      ElementCompute,
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
      ElementN,
      ElementSum,
      ElementSoft,
      kAlignmentC,
      MatrixShape<1, 1024>>;

 public:
  /// Arguments class
  struct Arguments {
    typename GemmKernel::Arguments gemm;

    typename SoftmaxApplyKernel::Arguments softmax;

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
        TensorRefSoft ref_Softmax_,
        int64_t batch_stride_A_ = 0,
        int64_t batch_stride_B_ = 0,
        int64_t batch_stride_C_ = 0,
        int64_t batch_stride_D_ = 0,
        int64_t batch_stride_Max_ = 0,
        int64_t batch_stride_Softmax_ = 0)
        : gemm(
              cutlass::gemm::GemmUniversalMode::kBatched,
              problem_size,
              batch_count_,
              ref_A_,
              ref_B_,
              batch_stride_A_,
              batch_stride_B_,
              typename EpilogueVisitor::Arguments(
                  linear_scaling,
                  ref_C_,
                  ref_D_,
                  ref_N_.data(),
                  batch_stride_C_,
                  batch_stride_D_,
                  batch_stride_Max_)),
          softmax(
              MatrixCoord(problem_size.m(), problem_size.n()),
              batch_count_,
              ref_D_,
              ref_N_,
              ref_Softmax_,
              batch_stride_D_,
              batch_stride_Max_,
              batch_stride_Softmax_) {}
  };

  struct Params {
    typename GemmKernel::Params gemm;

    typename SoftmaxApplyKernel::Params softmax;

    //
    // Methods
    //
    Params() {}

    Params(Arguments const& args) : gemm(args.gemm), softmax(args.softmax) {}
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

    cutlass::Kernel<GemmKernel>
        <<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the SoftmaxApplyKernel
    //

    dim3 apply_block(
        SoftmaxApplyKernel::Shape::kColumn, SoftmaxApplyKernel::Shape::kRow);

    int cta_rows = SoftmaxApplyKernel::Shape::kRow;
    int cta_columns =
        SoftmaxApplyKernel::Shape::kColumn * SoftmaxApplyKernel::kAlignment;

    dim3 apply_grid(
        (params_.softmax.args.extent.row() + cta_rows - 1) / cta_rows,
        (params_.softmax.args.extent.column() + cta_columns - 1) / cta_columns,
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
