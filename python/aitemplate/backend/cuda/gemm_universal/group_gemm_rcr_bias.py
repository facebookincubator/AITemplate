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
Codegen functions for group_gemm_rcr_bias.
"""
from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal import (
    common,
    group_common_bias,
    group_gemm_rcr,
)

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


@registry.reg("cuda.group_gemm_rcr_bias.config")
def group_rcr_config(func_attrs):
    group_gemm_rcr.group_rcr_config(func_attrs)


@registry.reg("cuda.group_gemm_rcr_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, shape_template):
    return group_common_bias.gen_profiler(
        func_attrs, workdir, profiler_filename, shape_template
    )


@registry.reg("cuda.group_gemm_rcr_bias.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
):
    return group_common_bias.gen_function(
        func_attrs,
        exec_cond_template,
        shape_eval_template,
    )


@registry.reg("cuda.group_gemm_rcr_bias.func_decl")
def gen_function_decl(func_attrs):
    return group_common_bias.gen_function_decl(func_attrs)


@registry.reg("cuda.group_gemm_rcr_bias.func_call")
def gen_function_call(func_attrs, indent="  "):
    return group_common_bias.gen_function_call(func_attrs, indent)


@registry.reg("cuda.group_gemm_rcr_bias.filter")
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
