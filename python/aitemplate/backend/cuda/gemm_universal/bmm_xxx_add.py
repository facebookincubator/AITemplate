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
Codegen for 8 bmm_xxx_add ops, which compute A @ B + bias + C. The ops differ
in layouts of A, B, and C/bias: each can be column-major or row-major,
8 combinations in total.

This module registers functions config, gen_profiler, gen_function, func_decl,
func_call, and filter for each layout combination under names like
"cuda.bmm_rcr_add.func_call".
"""


from aitemplate.backend import registry
from aitemplate.backend.common import gemm_common
from aitemplate.backend.cuda.gemm_universal import bmm_common, common
from aitemplate.backend.cuda.gemm_universal.bmm_xxx import _get_problem_args, get_config


def get_gen_function(a_layout, b_layout, c_layout):
    """
    Return gen_function for given layouts of A, B, and C/bias.
    """

    def gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
    ):
        problem_args = _get_problem_args(a_layout, b_layout, c_layout)
        default_mm_info = bmm_common.get_default_problem_info(
            problem_args,
            bias_ptr="d_ptr",
            alpha_value=func_attrs.get("alpha", 1),
            beta_value=1,
        )
        (
            problem_args,
            input_addr_calculator,
            output_addr_calculator,
        ) = bmm_common.make_function_strided_args(
            func_attrs, dim_info_dict, default_mm_info, is_permute=False
        )
        return bmm_common.gen_function(
            func_attrs,
            exec_cond_template,
            problem_args,
            dim_info_dict,
            input_addr_calculator=input_addr_calculator,
            output_addr_calculator=output_addr_calculator,
        )

    return gen_function


def get_gen_profiler(a_layout, b_layout, c_layout):
    """
    Return gen_profiler for given layouts of A, B, and C/bias.
    """

    def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
        a_dims = bmm_common.reverse_dim_info_mapping(
            dim_info_dict, gemm_common.Source.INPUT, 0
        )
        b_dims = bmm_common.reverse_dim_info_mapping(
            dim_info_dict, gemm_common.Source.INPUT, 1
        )
        c_dims = bmm_common.reverse_dim_info_mapping(
            dim_info_dict, gemm_common.Source.OUTPUT, 0
        )

        args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
            a_dims=a_dims, b_dims=b_dims, c_dims=c_dims
        )

        problem_args = _get_problem_args(a_layout, b_layout, c_layout)
        default_mm_info = bmm_common.get_default_problem_info(
            problem_args,
            bias_ptr="d_ptr",
            alpha_value=func_attrs.get("alpha", 1),
            beta_value=1,
        )
        a_shapes = func_attrs["input_accessors"][0].original_shapes
        b_shapes = func_attrs["input_accessors"][1].original_shapes
        d_shapes = func_attrs["input_accessors"][2].original_shapes
        bmm_common._update_stride_info(default_mm_info, a_shapes, b_shapes, d_shapes)

        problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
            mm_info=default_mm_info,
        )

        return bmm_common.gen_profiler(
            func_attrs,
            workdir,
            profiler_filename,
            dim_info_dict,
            common.SRC_TEMPLATE,
            problem_args,
            args_parser,
        )

    return gen_profiler


# Register functions for each of 8 layout combinations
for a_layout in ["c", "r"]:
    for b_layout in ["c", "r"]:
        for c_layout in ["c", "r"]:
            prefix = f"cuda.bmm_{a_layout}{b_layout}{c_layout}_add."

            config = get_config(a_layout, b_layout, c_layout)
            registry.reg(prefix + "config")(config)

            gen_profiler = get_gen_profiler(a_layout, b_layout, c_layout)
            registry.reg(prefix + "gen_profiler")(gen_profiler)

            gen_function = get_gen_function(a_layout, b_layout, c_layout)
            registry.reg(prefix + "gen_function")(gen_function)

            # The remaining 3 functions don't depend on the layout
            registry.reg(prefix + "func_decl")(bmm_common.gen_function_decl)
            registry.reg(prefix + "func_call")(bmm_common.gen_function_call)
            registry.reg(prefix + "filter")(common.function_filter)
