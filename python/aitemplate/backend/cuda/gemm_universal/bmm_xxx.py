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

from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal import bmm_common, common

"""
Codegen for 8 bmm_xxx ops, which compute A @ B + bias. The ops differ in
layouts of A, B, and bias: each can be column-major or row-major,
8 combinations in total.

This module registers functions config, gen_profiler, gen_function, func_decl,
func_call, and filter for each layout combination under names like
"cuda.bmm_rcr.func_call".
"""


def _get_problem_args(a_layout, b_layout, c_layout):
    return {
        "bias_ptr": "c_ptr",
        "a_batch_stride": "M * K",
        "b_batch_stride": "N * K",
        "bias_batch_stride": "M * N",
        "c_batch_stride": "M * N",
        "lda": "M" if a_layout == "c" else "K",
        "ldb": "K" if b_layout == "c" else "N",
        "ldbias": "M" if c_layout == "c" else "N",
        "ldc": "M" if c_layout == "c" else "N",
    }


def get_config(a_layout, b_layout, c_layout):
    """
    Return config function for given layouts of A, B, and bias.
    """

    def config(func_attrs, dtype="float16"):
        import cutlass_lib

        layout_choice = {
            "c": cutlass_lib.library.LayoutType.ColumnMajor,
            "r": cutlass_lib.library.LayoutType.RowMajor,
        }

        def fproc(op):
            return common.default_fproc(
                op=op,
                a_layout=layout_choice[a_layout],
                b_layout=layout_choice[b_layout],
                c_layout=layout_choice[c_layout],
                dtype=func_attrs["inputs"][0].dtype(),
                epilogue_name=func_attrs["epilogue"],
            )

        func_attrs["op_instance"] = common.extract_config(fproc)

    return config


def get_gen_profiler(a_layout, b_layout, c_layout):
    """
    Return gen_profiler for given layouts of A, B, and bias.
    """

    def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
        problem_args = _get_problem_args(a_layout, b_layout, c_layout)
        return bmm_common.default_gen_profiler(
            func_attrs,
            workdir,
            profiler_filename,
            dim_info_dict,
            problem_args,
        )

    return gen_profiler


def get_gen_function(a_layout, b_layout, c_layout):
    """
    Return gen_function for given layouts of A, B, and bias.
    """

    def gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
    ):
        problem_args = _get_problem_args(a_layout, b_layout, c_layout)

        default_mm_info = bmm_common.get_default_problem_info(
            problem_args,
            alpha_value=func_attrs.get("alpha", 1),
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


# Register functions for each of 8 layout combinations
for a_layout in ["c", "r"]:
    for b_layout in ["c", "r"]:
        for c_layout in ["c", "r"]:
            prefix = f"cuda.bmm_{a_layout}{b_layout}{c_layout}."

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
