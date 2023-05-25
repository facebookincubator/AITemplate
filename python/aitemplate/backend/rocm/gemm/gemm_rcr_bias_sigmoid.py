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
GEMM ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[m, n] = Sigmoid(a[m, k] * b[n, k] + bias[n])
This is used for `torch.nn.functional.linear + sigmoid`
When used for `linear`, need to set A->Data, B->Weight, C->Bias
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.gemm import common
from aitemplate.backend.rocm.gemm.layout import RCR

# pylint: disable=C0415,W0613

EXTRA_CODE = jinja2.Template(
    """
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {
namespace {
struct AddSigmoid
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = 1.0f / (1.0f + exp(-a));
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        const double a = x0 + x1;
        y              = 1.0 / (1.0 + exp(-a));
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t a = x0 + x1;
        y              = type_convert<half_t>(1.0) / (type_convert<half_t>(1.0) + type_convert<half_t>(exp(ck::type_convert<float>(-a))));
    };
};
} // namespace
} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
"""
)


@registry.reg("rocm.gemm_rcr_bias_sigmoid.config")
def gemm_config(func_attrs, dtype="float16"):
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
    extra_kind = ck_lib.library.TensorOperation.AddSigmoid
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.gemm_rcr_bias_sigmoid.gen_profiler")
def gemm_gen_profiler(func_attrs, workdir, dim_info_dict):
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
        gemm_flag="bias_sigmoid",
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("rocm.gemm_rcr_bias_sigmoid.gen_function")
def gemm_gen_function(func_attrs, exec_cond_template, dim_info_dict):
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
        "bias_sigmoid",
        extra_code=EXTRA_CODE.render(),
        input_addr_calculator=common.INPUT_ADDR_CALCULATOR.render(
            accessor_a=func_attrs["input_accessors"][0],
            accessor_b=func_attrs["input_accessors"][1],
        ),
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            output_accessor=func_attrs["output_accessors"][0]
        ),
    )


@registry.reg("rocm.gemm_rcr_bias_sigmoid.func_decl")
def gemm_gen_function_decl(func_attrs):
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
    return common.gen_function_decl(func_name=func_name, gemm_flag="bias_sigmoid")


@registry.reg("rocm.gemm_rcr_bias_sigmoid.func_call")
def gemm_gen_function_call(func_attrs, indent="  "):
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
    return common.gen_function_call(func_attrs, indent, gemm_flag="bias_sigmoid")


@registry.reg("rocm.gemm_rcr_bias_sigmoid.filter")
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
