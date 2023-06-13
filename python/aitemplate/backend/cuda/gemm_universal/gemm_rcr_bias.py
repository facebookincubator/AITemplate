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
C = GeMM(A, B) + bias
where A[RowMajor][M, K], B[ColMajor][N, K], bias[RowMajor][N]
"""
import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common, common_bias, gemm_rcr
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


EXTRA_CODE = jinja2.Template(
    """
using elem_input_type = {{elem_input_type}};
using elem_output_type = {{elem_output_type}};
"""
)


# used for real execution
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,          // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,          // void const * ptr_B
    ({{elem_input_type}}*)(bias_ptr),                        // void const * ptr_C
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // void * ptr_D
    input_a_batch_stride,                                    // int64_t batch_stride_A
    input_b_batch_stride,                                    // int64_t batch_stride_B
    N,                                                       // int64_t batch_stride_C
    M * N,                                                   // int64_t batch_stride_D
    input_a_stride,                                          // typename LayoutA::Stride::LongIndex lda
    input_b_stride,                                          // typename LayoutB::Stride::LongIndex ldb
    0,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
"""
)


# in case of TMA epilogue schedule, use the transposed problem to pass the
# column-major bias vector through the bias + elementwise epilogue (not residual)
PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
{% if has_tma_epilogue %}
    {
        static_cast<coord_t>(N),
        static_cast<coord_t>(M),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,              // ElementA const* ptr_A
    {input_b_stride, cute::Int<1>{}, cute::Int<0>{}},            // StrideA dA
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,              // ElementB const* ptr_B
    {input_a_stride, cute::Int<1>{}, cute::Int<0>{}},            // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename ThreadEpilogueOp::Params thread
        nullptr,                                                 // ElementC const* ptr_C
        {cute::Int<1>{}, cute::Int<0>{}, cute::Int<0>{}},        // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD const* ptr_D
        {cute::Int<1>{}, output_stride, cute::Int<0>{}},         // StrideD dD
        ({{elem_input_type}}*)(bias_ptr),                        // ElementBias const* ptr_Bias
    },                                                           // EpilogueArguments epilogue
{% else %}
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,              // ElementA const* ptr_A
    {input_a_stride, cute::Int<1>{}, cute::Int<0>{}},            // StrideA dA
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,              // ElementB const* ptr_B
    {input_b_stride, cute::Int<1>{}, cute::Int<0>{}},            // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename ThreadEpilogueOp::Params thread
        ({{elem_input_type}}*)(bias_ptr),                        // ElementC const* ptr_C
        {cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}},        // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD const* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
{% endif %}
"""
)


# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*)(a_ptr),                           // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr),                           // void const * ptr_B
    ({{elem_input_type}}*)(bias_ptr),                        // void const * ptr_C
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // void * ptr_D
    M * K,                                                   // int64_t batch_stride_A
    N * K,                                                   // int64_t batch_stride_B
    N,                                                       // int64_t batch_stride_C
    M * N,                                                   // int64_t batch_stride_D
    K,                                                       // typename LayoutA::Stride::LongIndex lda
    K,                                                       // typename LayoutB::Stride::LongIndex ldb
    0,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
"""
)


# in case of TMA epilogue schedule, use the transposed problem to pass the
# column-major bias vector through the bias + elementwise epilogue (not residual)
PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
{% if has_tma_epilogue %}
    {
        static_cast<coord_t>(N),
        static_cast<coord_t>(M),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    ({{elem_input_type}}*)(b_ptr),                               // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideA dA
    ({{elem_input_type}}*)(a_ptr),                               // ElementB const* ptr_B
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename ThreadEpilogueOp::Params thread
        nullptr,                                                 // ElementC const* ptr_C
        {cute::Int<1>{}, cute::Int<0>{}, cute::Int<0>{}},        // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD const* ptr_D
        {cute::Int<1>{}, output_stride, cute::Int<0>{}},         // StrideD dD
        ({{elem_input_type}}*)(bias_ptr),                        // ElementBias const* ptr_Bias
    },                                                           // EpilogueArguments epilogue
{% else %}
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    ({{elem_input_type}}*)(a_ptr),                               // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideA dA
    ({{elem_input_type}}*)(b_ptr),                               // ElementB const* ptr_B
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename ThreadEpilogueOp::Params thread
        ({{elem_input_type}}*)(bias_ptr),                        // ElementC const* ptr_C
        {cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}},        // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD const* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
{% endif %}
"""
)


@registry.reg("cuda.gemm_rcr_bias.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    common.make_fproc(func_attrs, RCR, include_cutlass_3x_ops=True)

    import cutlass_lib

    for op in func_attrs["op_instance"].values():
        if common.has_tma_epilogue(op):
            # disable residual to leave more SMEM for the mainloop
            op.C.element = cutlass_lib.library.DataType.void

            # swap the output layout to the transposed problem
            op.C.layout = cutlass_lib.library.LayoutType.ColumnMajor
            op.D.layout = cutlass_lib.library.LayoutType.ColumnMajor

            # switch to a TMA epilogue with bias
            op.epilogue_schedule = (
                cutlass_lib.library.EpilogueScheduleBiasElementwiseMapping[
                    op.epilogue_schedule
                ]
            )


@registry.reg("cuda.gemm_rcr_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    extra_code = EXTRA_CODE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return gemm_rcr.common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_bias.SRC_TEMPLATE,
        problem_args_template=PROFILER_PROBLEM_ARGS_TEMPLATE,
        problem_args_template_cutlass_3x=PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_rcr_bias.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_addr_calculator = gemm_rcr.get_input_addr_calculator(func_attrs)
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
        has_tma_epilogue=any(
            common.has_tma_epilogue(func_attrs["op_instance"][exec_item.algo])
            for exec_item in func_attrs["exec_path"].values()
        ),
    )
    extra_code = EXTRA_CODE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common.gen_function(
        func_attrs=func_attrs,
        src_template=common_bias.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_rcr_bias.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("cuda.gemm_rcr_bias.func_call")
def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]
    return common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.gemm_rcr_bias.filter")
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
