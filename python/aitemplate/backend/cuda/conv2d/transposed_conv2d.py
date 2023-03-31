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
transposed conv2d op codegen
"""
from aitemplate.backend import registry
from aitemplate.backend.cuda.conv2d import common, common_transposed_conv2d as ctc

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("cuda.transposed_conv2d.config")
def transposed_conv2d_config(
    func_attrs,
    dtype="float16",
):
    func_attrs["op_instance"] = ctc.extract_config(
        func_attrs=func_attrs,
        dtype=dtype,
    )


@registry.reg("cuda.transposed_conv2d.gen_profiler")
def transposed_conv2d_gen_profiler(
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
        f_emit_instance=ctc.emit_instance,
        is_transpose=True,
        instance_name_base="DeviceConvBwdInstance",
    )


@registry.reg("cuda.transposed_conv2d.gen_function")
def transposed_conv2d_gen_function(
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
        f_emit_instance=ctc.emit_instance,
        is_transpose=True,
    )


@registry.reg("cuda.transposed_conv2d.func_decl")
def transposed_conv2d_func_decl(
    func_attrs,
):
    return common.gen_function_decl(
        func_attrs=func_attrs,
    )


@registry.reg("cuda.transposed_conv2d.func_call")
def transposed_conv2d_func_call(
    func_attrs,
    indent="  ",
):
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        is_transpose=True,
    )


@registry.reg("cuda.transposed_conv2d.filter")
def transposed_conv2d_filter(
    cfg,
    func_attrs,
    x_shape,
):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(
        cfg=cfg,
        func_attrs=func_attrs,
        x_shape=x_shape,
    )
