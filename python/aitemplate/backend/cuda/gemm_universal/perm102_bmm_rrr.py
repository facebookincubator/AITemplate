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
Codegen functions for perm102_bmm_rrr, which computes
C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row))
"""
from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import bmm_common, common
from aitemplate.backend.cuda.gemm_universal.perm102_bmm_rcr import (
    get_output_addr_calculator,
)

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_default_problem_info(**kwargs):
    problem_args = {
        "bias_ptr": "c_ptr",
        "a_batch_stride": "K",
        "b_batch_stride": "N * K",
        "bias_batch_stride": "N",
        "c_batch_stride": "N",
        "lda": "K * B",
        "ldb": "N",
        "ldbias": "N * B",
        "ldc": "N * B",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


# Currently only has output Tensor Accessor support.
def _get_strided_problem_info(func_attrs):
    backend_spec = CUDASpec()
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    return bmm_common.Bmm_problem_info(
        a_ptr="(a_ptr)",
        b_ptr="(b_ptr)",
        bias_ptr="(" + elem_output_type + "*)(c_ptr) + output_offset",
        c_ptr="(" + elem_output_type + "*)(c_ptr) + output_offset",
        alpha_value=func_attrs.get("alpha", 1),
        a_batch_stride="K",
        b_batch_stride="N * K",
        bias_batch_stride="output_batch_stride",
        c_batch_stride="output_batch_stride",
        lda="K * B",
        ldb="N",
        ldbias="output_stride",
        ldc="output_stride",
    )


@registry.reg("cuda.perm102_bmm_rrr.config")
def gemm_rrr_config(func_attrs, dtype="float16"):
    def fproc(op):
        import cutlass_lib

        return common.default_fproc(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.RowMajor,
            b_layout=cutlass_lib.library.LayoutType.RowMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            dtype=func_attrs["inputs"][0].dtype(),
            epilogue_name=func_attrs["epilogue"],
        )

    func_attrs["op_instance"] = common.extract_config(fproc)


@registry.reg("cuda.perm102_bmm_rrr.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=["M", "B", "K"], b_dims=["B", "K", "N"], c_dims=["M", "B", "N"]
    )

    mm_info = _get_default_problem_info(
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


@registry.reg("cuda.perm102_bmm_rrr.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    bmm_problem_info = _get_strided_problem_info(func_attrs)

    # broadcasting is not supported
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=bmm_problem_info,
    )

    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
        "",  # input_addr_calculator
        get_output_addr_calculator(func_attrs),
    )


@registry.reg("cuda.perm102_bmm_rrr.func_decl")
def gen_function_decl(func_attrs):
    return bmm_common.gen_function_decl(func_attrs)


@registry.reg("cuda.perm102_bmm_rrr.func_call")
def gen_function_call(func_attrs, indent="  "):
    return bmm_common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.perm102_bmm_rrr.filter")
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
