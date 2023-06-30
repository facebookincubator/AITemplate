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
Codegen for conv2d.
"""
from aitemplate.backend import registry
from aitemplate.backend.cuda.conv2d import common

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("cuda.conv2d.config")
def conv2d_config(func_attrs):
    """Populates conv2d cutlass configs into 'op_instance' field."""
    func_attrs["op_instance"] = common.extract_config(
        func_attrs=func_attrs,
        dtype=func_attrs["inputs"][0]._attrs["dtype"],
    )


@registry.reg("cuda.conv2d.gen_profiler")
def conv2d_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
):
    """Codegen for conv2d profiler."""
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
    )


@registry.reg("cuda.conv2d.gen_function")
def conv2d_gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """Codegen for conv2d function."""
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
    )


@registry.reg("cuda.conv2d.func_decl")
def conv2d_func_decl(
    func_attrs,
):
    """Codegen for conv2d function declaration."""
    return common.gen_function_decl(
        func_attrs=func_attrs,
    )


@registry.reg("cuda.conv2d.func_call")
def conv2d_func_call(
    func_attrs,
    indent="  ",
):
    """Codegen for conv2d function call."""
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
    )


@registry.reg("cuda.conv2d.filter")
def conv2d_filter(
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
