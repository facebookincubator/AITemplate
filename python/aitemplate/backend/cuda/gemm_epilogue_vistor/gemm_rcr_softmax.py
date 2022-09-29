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

from ... import registry
from ..gemm_universal import common
from ..gemm_universal.layout import RCR
from . import common_softmax

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
    /*
        A: M*K (RowMajor)
        B: N*K (ColumnMajor)
        C/D/sofmax: M*N (RowMajor)
        N: M*1 (RowMajor)
    */

        {M, N, K},
        1,
        {a_ptr, LayoutA(K)},
        {b_ptr, LayoutB(K)},
        {c_ptr, LayoutC(N)},
        {d_ptr, LayoutC(N)},
        {
            float(1.0),
            float(0.0)
        },
        {n_ptr, LayoutC(1)},
        {soft_ptr, LayoutC(N)}

"""
)


@registry.reg("cuda.gemm_rcr_softmax.config")
def gemm_rcr_softmax_config(func_attrs, dtype="float16"):
    common.make_fproc_f16(func_attrs, RCR)


def common_gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    src_template,
    problem_args_template,
    bias_ptr_arg=None,
    extra_code="",
):
    output_addr_calculator = common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        stride_dim="*b_dim0"
    )
    common_softmax.gen_profiler(
        func_attrs,
        workdir,
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


@registry.reg("cuda.gemm_rcr_softmax.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    return common_gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common_softmax.SRC_TEMPLATE,
        PROBLEM_ARGS_TEMPLATE,
    )


@registry.reg("cuda.gemm_rcr_softmax.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args_template=None,
):
    if problem_args_template is None:
        problem_args = PROBLEM_ARGS_TEMPLATE.render()
    else:
        problem_args = problem_args_template.render()
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_softmax.gen_function(
        func_attrs,
        common_softmax.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims,
        weight_ndims,
        dim_info_dict,
        emit_kernel=True,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
    )


@registry.reg("cuda.gemm_rcr_softmax.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_softmax.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("cuda.gemm_rcr_softmax.func_call")
def gen_function_call(func_attrs, indent="  "):
    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]

    tmp_c = func_attrs["inputs"][2]
    tmp_d = func_attrs["inputs"][3]
    tmp_n = func_attrs["inputs"][4]

    soft = func_attrs["outputs"][0]
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
    return common_softmax.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        has_bias=has_bias,
        c_ptr=tmp_c._attrs["name"],
        d_ptr=tmp_d._attrs["name"],
        n_ptr=tmp_n._attrs["name"],
        soft_ptr=soft._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
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
