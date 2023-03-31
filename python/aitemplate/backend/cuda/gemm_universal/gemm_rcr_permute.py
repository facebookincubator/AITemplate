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
C = permute(GeMM(A, B) + bias)
where A[RowMajor][M, K], B[ColMajor][N, K], bias[RowMajor][N]
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common, common_permute

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);

  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = N;
  int64_t b_dim1 = K;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    {M, N, K},                                               // GemmCoord problem_size
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
    K,                                                       // typename LayoutB::Stride::LongIndex ldb
    N,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
"""
)


@registry.reg("cuda.gemm_rcr_permute.config")
def gemm_rcr_permute_config(func_attrs, dtype="float16"):
    def fproc(op):
        import cutlass_lib

        return common.default_fproc(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.RowMajor,
            b_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            dtype=func_attrs["inputs"][0].dtype(),
            epilogue_name=func_attrs["epilogue"],
            permute_layout=func_attrs["layout"],
        )

    func_attrs["op_instance"] = common_permute.extract_config(fproc, func_attrs)


def common_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    bias_ptr_arg=None,
    extra_code="",
):
    output_addr_calculator = common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        stride_dim="*b_dim0"
    )
    return common_permute.gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        src_template,
        problem_args_template,
        ARGS_PARSER_TEMPLATE,
        emit_kernel=True,
        support_split_k=True,
        output_addr_calculator=output_addr_calculator,
        bias_ptr_arg=bias_ptr_arg,
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_rcr_permute.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict, extra_code=""):
    extra_code = f"{common_permute.EXTRA_CODE.render()}\n{extra_code}"
    return common_gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        common.SRC_TEMPLATE,
        PROBLEM_ARGS_TEMPLATE,
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_rcr_permute.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args_template=None,
    extra_code="",
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    if problem_args_template is None:
        problem_args = PROBLEM_ARGS_TEMPLATE.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        )
    else:
        problem_args = problem_args_template.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        )
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    extra_code = f"{common_permute.EXTRA_CODE.render()}\n{extra_code}"
    return common_permute.gen_function(
        func_attrs,
        common.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims,
        weight_ndims,
        output_ndims,
        dim_info_dict,
        emit_kernel=True,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_rcr_permute.func_decl")
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


@registry.reg("cuda.gemm_rcr_permute.func_call")
def gen_function_call(func_attrs, indent="  "):
    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]

    output = func_attrs["outputs"][0]
    has_bias = False
    adims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][0].original_shapes
    ]
    bdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][1].original_shapes
    ]
    cdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["output_accessors"][0].original_shapes
    ]
    return common.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        has_bias=has_bias,
        c_ptr=output._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )


@registry.reg("cuda.gemm_rcr_permute.filter")
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
