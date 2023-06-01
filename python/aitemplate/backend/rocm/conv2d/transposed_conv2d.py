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
ROCM codegen functions for transposed_conv2d.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.conv2d import common

# pylint: disable=C0103,C0415,W0613
EXTRA_CODE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_xdl.hpp"
"""
)
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                N_,
{{indent}}                                K_,
{{indent}}                                C_,
{{indent}}                                {static_cast<ck::index_t>(*out_h), static_cast<ck::index_t>(*out_w)},
{{indent}}                                {static_cast<ck::index_t>(*kernel_h), static_cast<ck::index_t>(*kernel_w)},
{{indent}}                                {static_cast<ck::index_t>(*in_h), static_cast<ck::index_t>(*in_w)},
{{indent}}                                {static_cast<ck::index_t>(stride), static_cast<ck::index_t>(stride)},
{{indent}}                                {static_cast<ck::index_t>(dilation), static_cast<ck::index_t>(dilation)},
{{indent}}                                {static_cast<ck::index_t>(pad), static_cast<ck::index_t>(pad)},
{{indent}}                                {static_cast<ck::index_t>(pad), static_cast<ck::index_t>(pad)},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
"""
)


@registry.reg("rocm.transposed_conv2d.config")
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

    op_kind = ck_lib.library.Conv2dKind.TransposedConv2d
    extra_kind = ck_lib.library.TensorOperation.PassThrough
    func_attrs["op_instance"] = common.extract_config(op_kind, extra_kind)


@registry.reg("rocm.transposed_conv2d.gen_profiler")
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
        conv2d_flag="",
        prob_args_template=PROBLEM_ARGS_TEMPLATE,
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("rocm.transposed_conv2d.gen_function")
def conv2d_gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    instance_template : jinja2.Template
        Template that defines the model. e.g. 'using model=xxx'.
    exec_template : jinja2.Template
        Execution statements in main function.
    src_template : jinja2.Template
        Full main.cpp with headers, embedding all templates.
    exec_cond_template : jinja2.Template
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
        exec_cond_template,
        shape_eval_template,
        shape_save_template,
        "",
        prob_args_template=PROBLEM_ARGS_TEMPLATE,
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("rocm.transposed_conv2d.func_decl")
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
    return common.gen_function_decl(func_name=func_name, conv2d_flag="")


@registry.reg("rocm.transposed_conv2d.func_call")
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
    return common.gen_function_call(func_attrs, indent, conv2d_flag="")


@registry.reg("rocm.transposed_conv2d.filter")
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
