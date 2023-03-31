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
specialize conv2d op with few channels(< 8)
"""

from aitemplate.backend import registry
from aitemplate.backend.cuda.conv2d import (
    common,
    common_conv2d_bias_activation as cba,
    common_conv2d_few_channels as cfc,
)

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("cuda.conv2d_bias_hardswish_few_channels.config")
def conv2d_bias_hardswish_few_channels_config(
    func_attrs,
    dtype="float16",
):
    """extract configurations for profiling

    Parameters
    ----------
    func_attrs : Dict
        op attributes
    dtype : str, optional
        by default "float16"

    Returns
    -------
    None
    """
    func_attrs["op_instance"] = cfc.extract_config(
        func_attrs=func_attrs,
        dtype=dtype,
    )


@registry.reg("cuda.conv2d_bias_hardswish_few_channels.gen_profiler")
def conv2d_bias_hardswish_few_channels_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
):
    """generate code for profiling"""
    return cba.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
    )


@registry.reg("cuda.conv2d_bias_hardswish_few_channels.gen_function")
def conv2d_bias_hardswish_few_channels_gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """generating special conv2d kernel and all of its auxiliary functions

    Parameters
    ----------
    func_attrs : Dict
        [description] attributes of conv2d op
    exec_cond_template : [type]
        [description]
    shape_eval_template : [type]
        [description]
    shape_save_template : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return cba.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
    )


@registry.reg("cuda.conv2d_bias_hardswish_few_channels.func_decl")
def conv2d_bias_hardswish_few_channels_func_decl(
    func_attrs,
):
    return cba.gen_function_decl(
        func_attrs=func_attrs,
    )


@registry.reg("cuda.conv2d_bias_hardswish_few_channels.func_call")
def conv2d_bias_hardswish_few_channels_func_call(
    func_attrs,
    indent="  ",
):
    return cba.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
    )


@registry.reg("cuda.conv2d_bias_hardswish_few_channels.filter")
def conv2d_bias_hardswish_few_channels_filter(
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
