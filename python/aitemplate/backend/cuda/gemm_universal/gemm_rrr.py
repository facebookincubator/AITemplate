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
GEMM Specialization for
C = GeMM(A, B)
where A[RowMajor][M, K], B[RowMajor][K, N]
"""
import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common
from aitemplate.backend.cuda.gemm_universal.layout import RRR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);

  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = K;
  int64_t b_dim1 = N;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*)(a_ptr),                           // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr),                           // void const * ptr_B
    ({{elem_output_type}}*)(c_ptr),                          // void const * ptr_C
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // void * ptr_D
    M * K,                                                   // int64_t batch_stride_A
    N * K,                                                   // int64_t batch_stride_B
    M * N,                                                   // int64_t batch_stride_C
    M * N,                                                   // int64_t batch_stride_D
    K,                                                       // typename LayoutA::Stride::LongIndex lda
    N,                                                       // typename LayoutB::Stride::LongIndex ldb
    N,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
"""
)


PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    ({{elem_input_type}}*)(a_ptr),                               // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideA dA
    ({{elem_input_type}}*)(b_ptr),                               // ElementB const* ptr_B
    {cute::Int<1>{}, N, cute::Int<0>{}},                         // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename ThreadEpilogueOp::Params thread
        nullptr,                                                 // ElementC const* ptr_C
        {N, cute::Int<1>{}, cute::Int<0>{}},                     // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD const* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
"""
)


@registry.reg("cuda.gemm_rrr.config")
def gemm_rrr_config(func_attrs, dtype="float16"):
    common.make_fproc(func_attrs, RRR, include_cutlass_3x_ops=True)

    import cutlass_lib

    for op in func_attrs["op_instance"].values():
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            # disable residual to leave more SMEM for the mainloop
            op.C.element = cutlass_lib.library.DataType.void


@registry.reg("cuda.gemm_rrr.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    output_addr_calculator = common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        stride_dim="N"
    )
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common.SRC_TEMPLATE,
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        problem_args_template_cutlass_3x=PROBLEM_ARGS_TEMPLATE_CUTLASS_3X,
        args_parser_template=ARGS_PARSER_TEMPLATE,
        support_split_k=True,
        output_addr_calculator=output_addr_calculator,
    )


@registry.reg("cuda.gemm_rrr.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    problem_args_cutlass_3x = PROBLEM_ARGS_TEMPLATE_CUTLASS_3X.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common.gen_function(
        func_attrs=func_attrs,
        src_template=common.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="*b_dim1", output_accessor=func_attrs["output_accessors"][0]
        ),
    )


@registry.reg("cuda.gemm_rrr.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("cuda.gemm_rrr.func_call")
def gen_function_call(func_attrs, indent="  "):
    return common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.gemm_rrr.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
