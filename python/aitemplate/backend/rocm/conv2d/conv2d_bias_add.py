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
conv2d bias add codegen
"""
from ... import registry
from . import common

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("rocm.conv2d_bias_add_identity.config")
def conv2d_config(func_attrs, dtype="float16"):
    import ck_lib

    op_kind = ck_lib.library.Conv2dKind.GroupConv2dBiasRelu
    extra_kind = ck_lib.library.TensorOperation.AddAdd
    func_attrs["op_instance"] = common.extract_config(op_kind, extra_kind)


@registry.reg("rocm.conv2d_bias_add_identity.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        shape_template=shape_template,
        conv2d_flag="bias_add_identity",
        extra_code=common.HEADER_CODE.render(),
    )


@registry.reg("rocm.conv2d_bias_add_identity.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    return common.gen_function(
        func_attrs,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
        "bias_add_identity",
        common.HEADER_CODE.render(),
    )


@registry.reg("rocm.conv2d_bias_add_identity.func_decl")
def conv2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return common.gen_function_decl(
        func_name=func_name, conv2d_flag="bias_add_identity"
    )


@registry.reg("rocm.conv2d_bias_add_identity.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    return common.gen_function_call(func_attrs, indent, conv2d_flag="bias_add_identity")


@registry.reg("rocm.conv2d_bias_add_identity.filter")
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
