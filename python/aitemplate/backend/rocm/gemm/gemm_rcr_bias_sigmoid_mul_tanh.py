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
GEMM Specialization for
C = Tanh(Mul(Sigmoid(GeMM(A, B) + bias), D0)),
where A[RowMajor][M, K], B[ColMajor][N, K], C[RowMajor][M, N]
bias[RowMajor][N], D0[RowMajor][M, N].
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.gemm import common
from aitemplate.backend.rocm.gemm.layout import RCR


EXTRA_CODE = jinja2.Template(
    """
#include "data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {
namespace {
struct AddSigmoidMulTanh
{
    AddSigmoidMulTanh(){};

    __host__ __device__ void operator()(ck::half_t& e, const ck::half_t& c, const ck::half_t& bias, const ck::half_t& d0) const
    {
        const ck::half_t x = c + bias;
        // 1/(1+e^(-x))
        const ck::half_t x2 = type_convert<half_t>(1.0) / (type_convert<half_t>(1.0) + type_convert<half_t>(exp(ck::type_convert<float>(-x))));
        const ck::half_t x3 = x2*d0;
        // 1-2/(e^(2x)+1)
        e = type_convert<half_t>(1.0) - type_convert<half_t>(2.0) /
                    (type_convert<half_t>(1.0) + type_convert<half_t>(exp(ck::type_convert<float>(2*x3))));
    };
};
} // namespace
} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
"""
)


@registry.reg("rocm.gemm_rcr_bias_sigmoid_mul_tanh.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
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
    import ck_lib  # noqa: F401

    op_kind = ck_lib.library.GemmKind.Gemm
    extra_kind = ck_lib.library.TensorOperation.AddSigmoidMulTanh
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.gemm_rcr_bias_sigmoid_mul_tanh.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    """
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        dim_info_dict=dim_info_dict,
        args_parse=RCR.args_parse,
        gemm_flag="bias_sigmoid_mul_tanh",
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("rocm.gemm_rcr_bias_sigmoid_mul_tanh.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        "bias_sigmoid_mul_tanh",
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("rocm.gemm_rcr_bias_sigmoid_mul_tanh.func_decl")
def gen_function_decl(func_attrs):
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
    return common.gen_function_decl(
        func_name=func_name,
        gemm_flag="bias_sigmoid_mul_tanh",
        has_d0=common.has_d0(func_attrs),
        has_d1=common.has_d1(func_attrs),
    )


@registry.reg("rocm.gemm_rcr_bias_sigmoid_mul_tanh.func_call")
def gen_function_call(func_attrs, indent="  "):
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
    return common.gen_function_call(
        func_attrs, indent, gemm_flag="bias_sigmoid_mul_tanh"
    )


@registry.reg("rocm.gemm_rcr_bias_sigmoid_mul_tanh.filter")
def gemm_function_filter(cfg, func_attrs, x_shape):
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
