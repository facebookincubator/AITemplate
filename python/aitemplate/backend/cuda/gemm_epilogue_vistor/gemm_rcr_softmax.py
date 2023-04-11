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
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_epilogue_vistor import common_softmax
from aitemplate.backend.cuda.gemm_universal import common
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);

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
    /*
        A: (M, K) (RowMajor)
        B: (N, K) (ColumnMajor)
        C, D, Soft: (M, N) (RowMajor)
        N, S: (block_num, M) (RowMajor)
    */

    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                                                                                             // cutlass::gemm::GemmCoord problem_size
    1,                                                                                                                             // int32_t batch_count_
    {reinterpret_cast<{{elem_input_type}}*>(a_ptr), LayoutA(K)},                                                                   // TensorRefA ref_A_
    {reinterpret_cast<{{elem_input_type}}*>(b_ptr), LayoutB(K)},                                                                   // TensorRefB ref_B_
    {reinterpret_cast<{{elem_output_type}}*>(workspace), LayoutC(N)},                                                              // TensorRefC ref_C_
    {reinterpret_cast<{{elem_output_type}}*>(workspace + M * N * sizeof({{elem_output_type}})), LayoutC(N)},                       // TensorRefC ref_D_
    {
        float(1.0),
        float(0.0)
    },                                                                                                                             // typename EpilogueFunctorOp::Params linear_scaling
    {reinterpret_cast<float*>(workspace + 2 * M * N * sizeof({{elem_output_type}})), LayoutC(1)},                                  // TensorRefN ref_N_
    {reinterpret_cast<float*>(workspace + 2 * M * N * sizeof({{elem_output_type}}) + M * block_num * sizeof(float)), LayoutC(1)},  // TensorRefSum ref_S_
    {reinterpret_cast<{{elem_output_type}}*>(soft_ptr) + output_offset, LayoutC(output_stride)},                                   // TensorRefSoft ref_Softmax_
"""
)


@registry.reg("cuda.gemm_rcr_softmax.config")
def gemm_rcr_softmax_config(func_attrs, dtype="float16"):
    common.make_fproc(func_attrs, RCR)


def common_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    args_parser_template,
    **kwargs,
):
    output_addr_calculator = common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        stride_dim="*b_dim0"
    )

    return common_softmax.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=src_template,
        problem_args_template=problem_args_template,
        args_parser_template=args_parser_template,
        emit_kernel=True,
        output_addr_calculator=output_addr_calculator,
        **kwargs,
    )


@registry.reg("cuda.gemm_rcr_softmax.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
):
    return common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_softmax.SRC_TEMPLATE,
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        args_parser_template=ARGS_PARSER_TEMPLATE,
    )


@registry.reg("cuda.gemm_rcr_softmax.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args_template=None,
    **kwargs,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    if problem_args_template is None:
        problem_args_template = PROBLEM_ARGS_TEMPLATE

    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)

    return common_softmax.gen_function(
        func_attrs=func_attrs,
        src_template=common_softmax.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args_template.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
        ),
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        dim_info_dict=dim_info_dict,
        emit_kernel=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
        **kwargs,
    )


@registry.reg("cuda.gemm_rcr_softmax.func_decl")
def gen_function_decl(
    func_attrs,
    **kwargs,
):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)

    return common_softmax.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        **kwargs,
    )


@registry.reg("cuda.gemm_rcr_softmax.func_call")
def gen_function_call(
    func_attrs,
    indent="  ",
    **kwargs,
):
    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    soft = func_attrs["outputs"][0]

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

    return common_softmax.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        soft_ptr=soft._attrs["name"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
        **kwargs,
    )


@registry.reg("cuda.gemm_rcr_softmax.filter")
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
