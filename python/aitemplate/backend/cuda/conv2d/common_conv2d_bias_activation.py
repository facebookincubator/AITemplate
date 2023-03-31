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
common functions for conv_bias_activation subgraph
"""

from aitemplate.backend.cuda.conv2d import common

# pylint: disable=C0103,C0301

EXTRA_HEADER = """
#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>
#include <cutlass/epilogue/thread/linear_combination_hardswish.h>
"""


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
):
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
        is_bias=True,
        extra_header=EXTRA_HEADER,
    )


def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
        is_bias=True,
        extra_header=EXTRA_HEADER,
    )


def gen_function_decl(
    func_attrs,
):
    return common.gen_function_decl(
        func_attrs=func_attrs,
        is_bias=True,
    )


def gen_function_call(
    func_attrs,
    indent="  ",
):
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        is_bias=True,
    )
