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
common functions for conv2d bias act residual add
"""

from aitemplate.backend.cuda.conv2d import common

# pylint: disable=C0301,C0103

EXTRA_HEADER = """
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_residual_block.h>
"""


def extract_config(
    func_attrs,
    dtype="float16",
    activation_op_name="Identity",
    binary_op_name="Plus",
    unary_op_name="Identity",
):
    def set_ops(func_attrs, op):
        import cutlass_lib

        op.activation_op = cutlass_lib.library.EpilogueMathName[activation_op_name]
        op.binary_op = cutlass_lib.library.EpilogueMathName[binary_op_name]
        op.unary_op = cutlass_lib.library.EpilogueMathName[unary_op_name]

        return op

    return common.extract_config(
        func_attrs=func_attrs,
        dtype=dtype,
        skip_simt_kernels=True,
        f_apply_special_config=set_ops,
    )


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
        is_bias_add=True,
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
        is_bias_add=True,
        extra_header=EXTRA_HEADER,
    )


def gen_function_decl(
    func_attrs,
):
    return common.gen_function_decl(
        func_attrs=func_attrs,
        is_bias_add=True,
    )


def gen_function_call(
    func_attrs,
    indent="  ",
):
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        is_bias_add=True,
    )
