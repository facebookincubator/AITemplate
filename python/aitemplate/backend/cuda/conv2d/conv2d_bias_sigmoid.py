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
conv2d bias sigmoid codegen
"""

from ... import registry
from . import common, common_conv2d_bias_activation as cba

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("cuda.conv2d_bias_sigmoid.config")
def conv2d_config(func_attrs, dtype="float16"):
    func_attrs["op_instance"] = common.extract_config(func_attrs)


@registry.reg("cuda.conv2d_bias_sigmoid.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    return cba.gen_profiler(func_attrs, workdir, shape_template)


@registry.reg("cuda.conv2d_bias_sigmoid.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    return common.gen_function(
        func_attrs,
        cba.INSTANCE_TEMPLATE,
        cba.EXEC_TEMPLATE,
        cba.SRC_TEMPLATE,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
    )


@registry.reg("cuda.conv2d_bias_sigmoid.func_decl")
def conv2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return cba.FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.conv2d_bias_sigmoid.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    return cba.gen_function_call(func_attrs, indent)


@registry.reg("cuda.conv2d_bias_sigmoid.filter")
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
    return common.function_filter(cfg, func_attrs, x_shape)
