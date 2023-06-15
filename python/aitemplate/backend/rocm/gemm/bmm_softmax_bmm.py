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
Batched Gemm ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[b, m, n] = bmm(a[b, m, k], b[b, n, k])
This is used for `ops.bmm_rcr`.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.gemm import bmm_common, common
from aitemplate.backend.rocm.gemm.layout import RCR

EXTRA_CODE = jinja2.Template(
    """
#include "ck/utility/data_type.hpp"

const ck::half_t alpha = {{scale}};

namespace ck {
namespace tensor_operation {
namespace element_wise {
namespace {
struct AttnMul
{
    AttnMul(){};

    __host__ __device__ void operator()(ck::half_t& e, const ck::half_t& c) const
    {
        float s = {{scale}};
        e = c * type_convert<half_t>(s);
    };


    __host__ __device__ void operator()(float& e, const float& c) const
    {
        float s = {{scale}};
        e = c * s;
    };

};
} //namespace
} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
"""
)

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """

{{indent}}const int64_t stride_a = *a_dim2; // K
{{indent}}const int64_t stride_b = *b_dim2; // K
{{indent}}const int64_t stride_b1 = *c_dim2; // O
{{indent}}const int64_t stride_c = *c_dim2; // O

"""
)

EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp"
"""
)

PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{{indent}}                                static_cast<ck::half_t *>(bias_ptr),
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                O,
{{indent}}                                B,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{{indent}}                                stride_b1,
{{indent}}                                stride_c,
{{indent}}                                M*K,
{{indent}}                                N*K,
{{indent}}                                N*O,
{{indent}}                                M*O,
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::ScaleAndResetNaNToMinusInfinity{alpha},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
"""
)


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t B = std::atoi(argv[1]);
  int64_t M = std::atoi(argv[2]);
  int64_t N = std::atoi(argv[3]);
  int64_t K = std::atoi(argv[4]);
  int64_t O = std::atoi(argv[5]);

  // B,M,K * B,N,K = B,M,N // RCR
  // B,M,N * B,N,O = B,M,O // RRR
  int64_t a_dim0 = B;
  int64_t a_dim1 = M;
  int64_t a_dim2 = K;
  int64_t b_dim0 = B;
  int64_t b_dim1 = N;
  int64_t b_dim2 = K;
  int64_t b1_dim0 = B;
  int64_t b1_dim1 = N;
  int64_t b1_dim2 = O;
  int64_t c_dim0 = B;
  int64_t c_dim1 = M;
  int64_t c_dim2 = O;
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = B*M*K;
  int64_t b_ptr_sz = B*N*K;
  int64_t b1_ptr_sz = B*N*O;
  int64_t c_ptr_sz = B*M*O;
  int64_t ptr_max_sz = std::max({a_ptr_sz, b_ptr_sz, c_ptr_sz});
  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_max_sz)));

  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // x: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // b0: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // y: index 2
{% if "bias" in gemm_flag %}
  memory_pool->AllocateHalfTensor(b1_ptr_sz, mem_pool_sz);  // b1: index 3
{% endif %}
"""
)


@registry.reg("rocm.bmm_softmax_bmm.config")
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

    op_kind = ck_lib.library.GemmKind.BatchGemmSoftmaxGemm
    extra_kind = ck_lib.library.TensorOperation.PassThrough
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.bmm_softmax_bmm.gen_profiler")
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
        gemm_flag="bias_b1",
        extra_header_template=EXTRA_HEADER_TEMPLATE,
        tensor_decl_template=TENSOR_DECL_TEMPLATE,
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        extra_shape_template=EXTRA_SHAPE_TEMPLATE,
        extra_code=EXTRA_CODE.render(scale=func_attrs["scale"]),
    )


@registry.reg("rocm.bmm_softmax_bmm.gen_function")
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
        "bias_b1",
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        extra_shape_template=EXTRA_SHAPE_TEMPLATE,
        extra_header_template=EXTRA_HEADER_TEMPLATE,
        extra_code=EXTRA_CODE.render(scale=func_attrs["scale"]),
    )


@registry.reg("rocm.bmm_softmax_bmm.func_decl")
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
    return bmm_common.gen_function_decl(func_name=func_name, gemm_flag="bias_b1")


@registry.reg("rocm.bmm_softmax_bmm.func_call")
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
    return bmm_common.gen_function_call(func_attrs, indent, gemm_flag="bias_b1")


@registry.reg("rocm.bmm_softmax_bmm.filter")
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
