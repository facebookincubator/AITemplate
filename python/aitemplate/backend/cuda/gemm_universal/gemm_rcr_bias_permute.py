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
GEMM with bias and permute epilogue fusion
"""

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import (
    common,
    common_bias,
    common_permute,
    gemm_rcr_bias,
    gemm_rcr_permute,
)

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

PROBLEM_ARGS_TEMPLATE = gemm_rcr_bias.PROFILER_PROBLEM_ARGS_TEMPLATE


@registry.reg("cuda.gemm_rcr_bias_permute.config")
def gemm_rcr_bias_permute_config(func_attrs, dtype="float16"):
    return gemm_rcr_permute.gemm_rcr_permute_config(func_attrs, dtype)


@registry.reg("cuda.gemm_rcr_bias_permute.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    return gemm_rcr_permute.common_gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        common_bias.SRC_TEMPLATE,
        PROBLEM_ARGS_TEMPLATE,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_bias_permute.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args_template=None,
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
    return common_permute.gen_function(
        func_attrs,
        common_bias.SRC_TEMPLATE,
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
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_bias_permute.func_decl")
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


@registry.reg("cuda.gemm_rcr_bias_permute.func_call")
def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]
    return common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.gemm_rcr_bias_permute.filter")
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
