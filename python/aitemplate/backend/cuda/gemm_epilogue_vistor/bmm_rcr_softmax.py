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
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from ... import registry
from ..gemm_universal import common
from ..gemm_universal.layout import RCR
from . import bmm_common_softmax as bmm_common, common_softmax

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


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
  int64_t b_dim1 = N;
  int64_t b_dim2 = K;
  int64_t c_dim0 = B;
  int64_t c_dim1 = M;
  int64_t c_dim2 = N;
"""
)

PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    /*
        A: B*M*K (RowMajor)
        B: B*N*K (ColumnMajor)
        C/D/sofmax: B*M*N (RowMajor)
        N: B*M*1 (RowMajor)
    */

        {M, N, K},
        B,
        {a_ptr, LayoutA(K)},
        {b_ptr, LayoutB(K)},
        {c_ptr, LayoutC(N)},
        {d_ptr, LayoutC(N)},
        {
            float(1.0),
            float(0.0)
        },
        {n_ptr, LayoutC(1)},
        {soft_ptr, LayoutC(N)},
        M*K,
        N*K,
        M*N,
        M*N,
        M*N,
        M*N


"""
)


@registry.reg("cuda.bmm_rcr_softmax.config")
def bmm_rcr_softmax_config(func_attrs, dtype="float16"):
    """This function sets a callback for processing the epilogue of the kernel
    associated with func_attrs.

    Parameters
    ----------
    func_attrs: Dictionary
        kernel attributes dictionary
    layout: layout object
        kernel layout
    Returns
    -------
    None
    """
    common.make_fproc_f16(func_attrs, RCR)


@registry.reg("cuda.bmm_rcr_softmax.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    """Generate code for profiling"""
    return bmm_common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common_softmax.SRC_TEMPLATE,
        PROBLEM_ARGS_TEMPLATE,
        ARGS_PARSER_TEMPLATE,
        emit_kernel=True,
    )


@registry.reg("cuda.bmm_rcr_softmax.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    """Generate the code for main function"""
    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        PROBLEM_ARGS_TEMPLATE.render(),
    )


@registry.reg("cuda.bmm_rcr_softmax.func_decl")
def gen_function_decl(func_attrs):
    """Rendering argument to function declaration template"""
    func_name = func_attrs["name"]
    return bmm_common.FUNC_DECL_TEMPLATE.render(func_name=func_name, ndims=3)


@registry.reg("cuda.bmm_rcr_softmax.func_call")
def gen_function_call(func_attrs, indent="  "):
    """Rendering the code to function call template"""
    return bmm_common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.bmm_rcr_softmax.filter")
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
