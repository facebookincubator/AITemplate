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
Codegen functions for group_gemm_rcr.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal import common, group_common
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    problem_sizes_device,                                    // GemmCoord *problem_sizes
    problem_count,                                           // int problem_count
    threadblock_count,                                       // int threadblock_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename EpilogueOutputOp::Params output_op
    ({{elem_input_type}}**)(ptr_A),                          // ElementA ** ptr_A
    ({{elem_input_type}}**)(ptr_B),                          // ElementB ** ptr_B
    ({{elem_output_type}}**)(ptr_C),                         // ElementC ** ptr_C
    ({{elem_output_type}}**)(ptr_C),                         // ElementC ** ptr_D
    lda,                                                     // typename LayoutA::Stride::LongIndex *lda
    ldb,                                                     // typename LayoutB::Stride::LongIndex *ldb
    ldc,                                                     // typename LayoutC::Stride::LongIndex *ldc
    ldc,                                                     // typename LayoutC::Stride::LongIndex *ldd
"""
)


@registry.reg("cuda.group_gemm_rcr.config")
def group_rcr_config(func_attrs):
    common.make_fproc(func_attrs, RCR)


@registry.reg("cuda.group_gemm_rcr.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, shape_template):
    return group_common.gen_profiler(
        func_attrs, workdir, profiler_filename, shape_template, PROBLEM_ARGS_TEMPLATE
    )


@registry.reg("cuda.group_gemm_rcr.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
):
    return group_common.gen_function(
        func_attrs,
        exec_cond_template,
        shape_eval_template,
        PROBLEM_ARGS_TEMPLATE,
    )


@registry.reg("cuda.group_gemm_rcr.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return group_common.FUNC_DECL_TEMPLATE.render(
        func_name=func_name, groups=func_attrs["groups"]
    )


@registry.reg("cuda.group_gemm_rcr.func_call")
def gen_function_call(func_attrs, indent="  "):
    ndims = 2
    return group_common.gen_function_call(func_attrs, ndims)


@registry.reg("cuda.group_gemm_rcr.filter")
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
