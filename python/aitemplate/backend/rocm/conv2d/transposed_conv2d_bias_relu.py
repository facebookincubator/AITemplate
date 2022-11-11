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
ROCM codegen functions for transposed_conv2d_bias_relu.
"""
import jinja2

from ... import registry
from . import common

# pylint: disable=C0103,C0415,W0613

EXTRA_CODE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"
"""
)


@registry.reg("rocm.transposed_conv2d_bias_relu.config")
def conv2d_config(func_attrs):
    """Extract (operation name, operation instance) pair from
    all operation candidates.


    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    Dict
        Extracted (operation name, operation instance) pair
        from all operation candidates.
    """
    import ck_lib

    op_kind = ck_lib.library.Conv2dKind.TransposedConv2dBiasRelu
    extra_kind = ck_lib.library.TensorOperation.AddRelu
    func_attrs["op_instance"] = common.extract_config(op_kind, extra_kind)


@registry.reg("rocm.transposed_conv2d_bias_relu.gen_profiler")
def conv2d_gen_profiler(func_attrs, workdir, shape_template):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    shape_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    """
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        shape_template=shape_template,
        conv2d_flag="bias_relu",
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("rocm.transposed_conv2d_bias_relu.gen_function")
def conv2d_gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_remplate : jinja2.Template
        Generates if statement to execute kernel.
    shape_eval_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    shape_save_template : jinja2.Template
        Generates output dimensions.
        The template is passed from compiler/ops/pool.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return common.gen_function(
        func_attrs,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
        "bias_relu",
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("rocm.transposed_conv2d_bias_relu.func_decl")
def conv2d_gen_function_decl(func_attrs):
    """Generates function declarations.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    str
        The rentered template of function declaration.
    """
    func_name = func_attrs["name"]
    return common.gen_function_decl(func_name=func_name, conv2d_flag="bias_relu")


@registry.reg("rocm.transposed_conv2d_bias_relu.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    return common.gen_function_call(func_attrs, indent, conv2d_flag="bias_relu")


@registry.reg("rocm.transposed_conv2d_bias_relu.filter")
def conv2d_function_filter(cfg, func_attrs, x_shape):
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
    return True
