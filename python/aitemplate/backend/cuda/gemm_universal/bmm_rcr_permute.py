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
Codegen for bmm_rcr_permute, which computes permute(A @ B + bias).
A[RowMajor], B[ColMajor], bias[RowMajor]
"""

from aitemplate.backend import registry
from aitemplate.backend.common import gemm_common
from aitemplate.backend.cuda.gemm_universal import (
    bmm_common,
    bmm_permute_common,
    common,
    common_permute,
)

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


PROBLEM_ARGS = {
    "bias_ptr": "c_ptr",
    "a_batch_stride": "M * K",
    "b_batch_stride": "N * K",
    "bias_batch_stride": "M * N",
    "c_batch_stride": "0",
    "lda": "K",
    "ldb": "K",
    "ldbias": "N",
    "ldc": "N",
}


@registry.reg("cuda.bmm_rcr_permute.config")
def bmm_rcr_permute_config(func_attrs, dtype="float16"):
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


@registry.reg("cuda.bmm_rcr_permute.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    a_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.INPUT, 0
    )
    b_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.INPUT, 1
    )
    c_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.OUTPUT, 0
    )

    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=a_dims, b_dims=b_dims, c_dims=c_dims
    )

    default_mm_info = bmm_common.get_default_problem_info(
        PROBLEM_ARGS,
        alpha_value=func_attrs.get("alpha", 1),
    )
    a_shapes = func_attrs["input_accessors"][0].original_shapes
    b_shapes = func_attrs["input_accessors"][1].original_shapes
    bmm_common._update_stride_info(default_mm_info, a_shapes, b_shapes)

    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=default_mm_info,
    )

    return bmm_permute_common.gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        common.SRC_TEMPLATE,
        problem_args,
        args_parser,
        emit_kernel=True,
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.bmm_rcr_permute.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    default_mm_info = bmm_common.get_default_problem_info(
        PROBLEM_ARGS,
        alpha_value=func_attrs.get("alpha", 1),
    )
    (
        problem_args,
        input_addr_calculator,
        output_addr_calculator,
    ) = bmm_common.make_function_strided_args(
        func_attrs, dim_info_dict, default_mm_info, is_permute=True
    )

    return bmm_permute_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
        input_addr_calculator,
        output_addr_calculator,
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.bmm_rcr_permute.func_decl")
def gen_function_decl(func_attrs):
    return bmm_permute_common.gen_function_decl(func_attrs)


@registry.reg("cuda.bmm_rcr_permute.func_call")
def gen_function_call(func_attrs, indent="  "):
    return bmm_permute_common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.bmm_rcr_permute.filter")
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
