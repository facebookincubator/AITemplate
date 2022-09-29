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
Common functions and templates for bmm_permute-family ops
"""
from ...common import gemm_common
from ..gemm_universal import common, common_bias

from . import bmm_common, common_permute

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    src_template,
    problem_args,
    args_parser,
    emit_kernel=False,
    bias_ptr_arg=None,
    extra_code="",
):
    """Generate code for profiling"""
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    has_d = False
    if "has_d" in func_attrs:
        has_d = func_attrs["has_d"]

    a_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    b_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    c_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    file_pairs = []
    has_bias = bias_ptr_arg is not None
    assert not (has_d and has_bias)
    for op_name, op in op_instance.items():
        config = common_permute.emit_instance(
            op,
            for_profiler=True,
            emit_kernel=emit_kernel,
            func_attrs=func_attrs,
        )
        config_name = common.extract_config_name(config)
        name = "GemmInstance"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=name, config=config
        )
        exec_program = common.EXEC_TEMPLATE.render(
            indent="  ",
            instance=name,
            is_profiler=True,
            problem_args=problem_args,
        )
        input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
            input_ndims=a_ndims,
            weight_ndims=b_ndims,
            output_ndims=c_ndims,
        )
        op_func = src_template.render(
            instances=instance,
            function_name="bmm",
            input_ndims=a_ndims,
            weight_ndims=b_ndims,
            output_ndims=c_ndims,
            shape_eval=shape_func,
            input_output_checks=input_output_checks,
            exec_paths=exec_program,
            has_d=has_d,
            extra_code=extra_code,
        )
        a_dims_ptr = [f"&a_dim{idx}" for idx in range(a_ndims)]
        b_dims_ptr = [f"&b_dim{idx}" for idx in range(b_ndims)]
        c_dims_ptr = [f"&c_dim{idx}" for idx in range(c_ndims)]
        func_call = bmm_common.FUNC_CALL_TEMPLATE.render(
            func_name="bmm",
            a_ptr="memory_pool->RequestHalfTensorByIdx(0)",
            b_ptr="memory_pool->RequestHalfTensorByIdx(1)",
            has_bias=has_bias,
            bias_ptr=bias_ptr_arg,
            c_ptr="memory_pool->RequestHalfTensorByIdx(2)",
            d_ptr="memory_pool->RequestHalfTensorByIdx(%d)" % (4 if has_bias else 3),
            has_d=has_d,
            a_dims_ptr=a_dims_ptr,
            b_dims_ptr=b_dims_ptr,
            c_dims_ptr=c_dims_ptr,
        )
        code = common.PROFILER_TEMPLATE.render(
            op_func=op_func,
            args_parse=args_parser,
            func_call=func_call,
            name=name,
            tensor_decl=bmm_common.TENSOR_DECL_TEMPLATE.render(
                name=name,
                a_ndims=a_ndims,
                b_ndims=b_ndims,
                c_ndims=c_ndims,
                has_d=has_d,
                has_bias=has_bias,
            ),
        )
        common.add_profiler(file_pairs, workdir, op_type, op_name, code)
    # build
    common.build_profiler(file_pairs)


def gen_function_decl(func_attrs):
    """Rendering argument to function declaration template"""
    func_name = func_attrs["name"]
    has_d = False
    if "has_d" in func_attrs:
        has_d = func_attrs["has_d"]
    return bmm_common.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        a_ndims=len(func_attrs["input_accessors"][0].original_shapes),
        b_ndims=len(func_attrs["input_accessors"][1].original_shapes),
        c_ndims=len(func_attrs["output_accessors"][0].original_shapes),
        has_d=has_d,
    )


def gen_function(
    func_attrs,
    exec_cond_template,
    problem_args,
    dim_info_dict,
    input_addr_calculator="",
    output_addr_calculator="",
    extra_code="",
    has_bias=False,
):
    return common_permute.gen_function(
        func_attrs,
        common_bias.SRC_TEMPLATE if has_bias else common.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims=len(func_attrs["input_accessors"][0].original_shapes),
        weight_ndims=len(func_attrs["input_accessors"][1].original_shapes),
        output_ndims=len(func_attrs["output_accessors"][0].original_shapes),
        dim_info_dict=dim_info_dict,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
        emit_kernel=True,
        extra_code=extra_code,
    )


def gen_function_call(func_attrs, indent="  ", bias_ptr_arg=None):
    return bmm_common.gen_function_call(func_attrs, indent, bias_ptr_arg)
