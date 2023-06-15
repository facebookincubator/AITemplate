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
Batched Gemm ROCM backend A[RowMajor], B[RowMajor], C[RowMajor], i.e.
c[b, m, n] = a[b, m, k] * b[b, k, n]
This is used for `ops.bmm_rrr`.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.gemm import bmm_common, common
from aitemplate.backend.rocm.gemm.layout import RRR


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t B = std::atoi(argv[1]);
  int64_t M = std::atoi(argv[2]);
  int64_t N = std::atoi(argv[3]);
  int64_t K = std::atoi(argv[4]);

  int64_t a_dim0 = B;
  int64_t a_dim1 = M;
  int64_t a_dim2 = K;
  int64_t b_dim0 = B;
  int64_t b_dim1 = K;
  int64_t b_dim2 = N;
  int64_t c_dim0 = B;
  int64_t c_dim1 = M;
  int64_t c_dim2 = N;
"""
)


@registry.reg("rocm.bmm_rrr.config")
def bmm_config(func_attrs, dtype="float16"):
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

    op_kind = ck_lib.library.GemmKind.BatchGemm
    extra_kind = ck_lib.library.TensorOperation.PassThrough
    common.make_fproc_f16(func_attrs, RRR, op_kind, extra_kind)


@registry.reg("rocm.bmm_rrr.gen_profiler")
def bmm_gen_profiler(func_attrs, workdir, dim_info_dict):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from bmm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    """
    return bmm_common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        args_parse=ARGS_PARSER_TEMPLATE.render(),
        dim_info_dict=dim_info_dict,
        gemm_flag="",
    )


@registry.reg("rocm.bmm_rrr.gen_function")
def bmm_gen_function(func_attrs, exec_cond_template, dim_info_dict):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from bmm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        "",
        input_addr_calculator=bmm_common.INPUT_ADDR_CALCULATOR.render(
            accessor_a=func_attrs["input_accessors"][0],
            accessor_b=func_attrs["input_accessors"][1],
        ),
        output_addr_calculator=bmm_common.OUTPUT_ADDR_CALCULATOR.render(
            output_accessor=func_attrs["output_accessors"][0]
        ),
    )


@registry.reg("rocm.bmm_rrr.func_decl")
def bmm_gen_function_decl(func_attrs):
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
    return bmm_common.gen_function_decl(func_name=func_name, gemm_flag="")


@registry.reg("rocm.bmm_rrr.func_call")
def bmm_gen_function_call(func_attrs, indent="  "):
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
    return bmm_common.gen_function_call(func_attrs, indent, gemm_flag="")


@registry.reg("rocm.bmm_rrr.filter")
def bmm_function_filter(cfg, func_attrs, x_shape):
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
    return True
