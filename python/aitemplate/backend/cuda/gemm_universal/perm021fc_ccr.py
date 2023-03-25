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
Codegen functions for perm021fc_ccr, which computes
[b, m, n] = bmm([b, k, m], [1, n, k]).
"""
from aitemplate.backend import registry
from aitemplate.backend.cuda.gemm_universal import bmm_common, common

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_problem_info(**kwargs):
    problem_args = {
        "bias_ptr": "c_ptr",
        "a_batch_stride": "M * K",
        "b_batch_stride": "0",
        "bias_batch_stride": "M * N",
        "c_batch_stride": "M * N",
        "lda": "M",
        "ldb": "K",
        "ldbias": "N",
        "ldc": "N",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


@registry.reg("cuda.perm021fc_ccr.config")
def gemm_ccr_config(func_attrs, dtype="float16"):
    def fproc(op):
        import cutlass_lib

        return common.default_fproc(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            b_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            dtype=func_attrs["inputs"][0].dtype(),
            epilogue_name=func_attrs["epilogue"],
        )

    func_attrs["op_instance"] = common.extract_config(fproc)


@registry.reg("cuda.perm021fc_ccr.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=["B", "K", "M"], b_dims=["1", "N", "K"], c_dims=["B", "M", "N"]
    )

    mm_info = _get_problem_info(
        alpha_value=func_attrs.get("alpha", 1),
    )
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=mm_info,
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


@registry.reg("cuda.perm021fc_ccr.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    mm_info = _get_problem_info(
        alpha_value=func_attrs.get("alpha", 1),
    )
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=mm_info,
    )

    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
    )


@registry.reg("cuda.perm021fc_ccr.func_decl")
def gen_function_decl(func_attrs):
    return bmm_common.gen_function_decl(func_attrs)


@registry.reg("cuda.perm021fc_ccr.func_call")
def gen_function_call(func_attrs, indent="  "):
    return bmm_common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.perm021fc_ccr.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
