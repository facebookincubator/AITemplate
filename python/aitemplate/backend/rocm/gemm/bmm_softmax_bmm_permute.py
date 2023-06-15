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
Batched Gemm ROCM backend for

  BMM + Softmax + BMM
  (B, M, K) * (B, N, K) = (B, M, N) #RCR
  softmax on dim N (B, M, N)
  (B, M, N) * (B, N, O) = (B, M, O) #RRR

This is used for `ops.bmm_softmax_bmm_permute`.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.common import gemm_common
from aitemplate.backend.rocm.gemm import bmm_common, common
from aitemplate.backend.rocm.gemm.layout import RCR

INPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  ck::index_t in_batch_stride = {{in_batch_stride_dim}};
  ck::index_t in_stride = {{in_stride_dim}};
  int64_t in_offset = {{in_offset_val}}; // default to 0
  ck::index_t weight_batch_stride = {{weight_batch_stride_dim}};
  ck::index_t weight_stride = {{weight_stride_dim}};
  int64_t weight_offset = {{weight_offset_val}}; // default to 0
  ck::index_t bias_batch_stride = {{bias_batch_stride_dim}};
  ck::index_t bias_stride = {{bias_stride_dim}};
  int64_t bias_offset = {{bias_offset_val}}; // default to 0
    """
)


EXTRA_CODE = jinja2.Template(
    """
const ck::half_t alpha = {{scale}};
"""
)

PROFILER_EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}const ck::index_t G1 = p_dim0; // G1

{{indent}}const ck::index_t in_batch_stride=M * K;
{{indent}}const ck::index_t in_stride=K;
{{indent}}const int64_t in_offset=0;
{{indent}}const ck::index_t weight_batch_stride=N * K;
{{indent}}const ck::index_t weight_stride=K;
{{indent}}const int64_t weight_offset=0;
{{indent}}const ck::index_t bias_batch_stride=N * O;
{{indent}}const ck::index_t bias_stride=O;
{{indent}}const int64_t bias_offset=0;

"""
)

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}const ck::index_t G1 = p_dim0; // G1
"""
)

EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t*>(in_ptr) + in_offset,
{{indent}}                                static_cast<ck::half_t*>(weight_ptr) + weight_offset,
{{indent}}                                static_cast<ck::half_t*>(bias_ptr) + bias_offset,
{{indent}}                                static_cast<ck::half_t*>(out_ptr),
{{indent}}                                {},
{{indent}}                                {},
{{indent}}                                {B/G1, G1, M, K},
{{indent}}                                {G1*in_batch_stride, in_batch_stride, in_stride, 1},
{{indent}}                                {B/G1, G1, N, K},
{{indent}}                                {G1*weight_batch_stride, weight_batch_stride, weight_stride, 1},
{{indent}}                                {B/G1, G1, O, N},
{{indent}}                                {G1*bias_batch_stride, bias_batch_stride, 1, bias_stride},
{{indent}}                                {B/G1, G1, M, O},
{{indent}}                                {M*G1*O, O, G1*O, 1},
{{indent}}                                {},
{{indent}}                                {},
{{indent}}                                {},
{{indent}}                                {},
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
  int64_t G = std::atoi(argv[6]);

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
  int64_t p_dim0 = G;
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


@registry.reg("rocm.bmm_softmax_bmm_permute.config")
def bmm_softmax_bmm_permute_config(func_attrs, dtype="float16"):
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

    op_kind = ck_lib.library.GemmKind.BatchGemmSoftmaxGemmPermute
    extra_kind = ck_lib.library.TensorOperation.PassThrough
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.bmm_softmax_bmm_permute_causal.config")
def bmm_softmax_bmm_permute_causal_config(func_attrs, dtype="float16"):
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

    op_kind = ck_lib.library.GemmKind.BatchGemmSoftmaxGemmPermute
    extra_kind = ck_lib.library.TensorOperation.CausalMask
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.bmm_softmax_bmm_permute.gen_profiler")
@registry.reg("rocm.bmm_softmax_bmm_permute_causal.gen_profiler")
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
        extra_shape_template=PROFILER_EXTRA_SHAPE_TEMPLATE,
        extra_code=EXTRA_CODE.render(scale=func_attrs["scale"]),
    )


@registry.reg("rocm.bmm_softmax_bmm_permute.gen_function")
@registry.reg("rocm.bmm_softmax_bmm_permute_causal.gen_function")
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
    in_batch_stride_dim = "M * K"
    in_stride_k_dim = "K"
    in_offset = 0
    weight_batch_stride_dim = "N * K"
    weight_stride_k_dim = "K"
    weight_offset = 0
    bias_batch_stride_dim = "N * O"
    bias_stride_k_dim = "O"
    bias_offset = 0

    if "input_accessors" in func_attrs:
        in_accessor = func_attrs["input_accessors"][0]
        weight_accessor = func_attrs["input_accessors"][1]
        bias_accessor = func_attrs["input_accessors"][2]

        if in_accessor.is_from_strided_tensor:
            in_offset = in_accessor.offset
            if not in_accessor.is_contiguous:
                a_dims = bmm_common.reverse_dim_info_mapping(
                    dim_info_dict, gemm_common.Source.INPUT, 0
                )

                in_batch_stride_dim = in_accessor.gen_stride_str(0, a_dims)
                in_stride_k_dim = in_accessor.stride(1)

        if weight_accessor.is_from_strided_tensor:
            weight_offset = weight_accessor.offset
            if not weight_accessor.is_contiguous:
                w_dims = bmm_common.reverse_dim_info_mapping(
                    dim_info_dict, gemm_common.Source.INPUT, 1
                )

                weight_batch_stride_dim = weight_accessor.gen_stride_str(0, w_dims)
                weight_stride_k_dim = weight_accessor.stride(1)

        if bias_accessor.is_from_strided_tensor:
            bias_offset = bias_accessor.offset
            if not bias_accessor.is_contiguous:
                b_dims = bmm_common.reverse_dim_info_mapping(
                    dim_info_dict, gemm_common.Source.INPUT, 2
                )

                bias_batch_stride_dim = bias_accessor.gen_stride_str(0, b_dims)
                bias_stride_k_dim = bias_accessor.stride(1)

    input_addr_calculator = INPUT_ADDR_CALCULATOR.render(
        in_batch_stride_dim=in_batch_stride_dim,
        in_stride_dim=in_stride_k_dim,
        in_offset_val=in_offset,
        weight_batch_stride_dim=weight_batch_stride_dim,
        weight_stride_dim=weight_stride_k_dim,
        weight_offset_val=weight_offset,
        bias_batch_stride_dim=bias_batch_stride_dim,
        bias_stride_dim=bias_stride_k_dim,
        bias_offset_val=bias_offset,
    )
    # TODO: add support for output_tensor_accessors

    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        "bias_b1",
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        extra_shape_template=EXTRA_SHAPE_TEMPLATE,
        extra_header_template=EXTRA_HEADER_TEMPLATE,
        extra_code=EXTRA_CODE.render(scale=func_attrs["scale"]),
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator="",
    )


@registry.reg("rocm.bmm_softmax_bmm_permute.func_decl")
@registry.reg("rocm.bmm_softmax_bmm_permute_causal.func_decl")
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
    return bmm_common.gen_function_decl(
        func_name=func_name, gemm_flag="bias_b1", pdims=len(func_attrs["shape"])
    )


@registry.reg("rocm.bmm_softmax_bmm_permute.func_call")
@registry.reg("rocm.bmm_softmax_bmm_permute_causal.func_call")
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


@registry.reg("rocm.bmm_softmax_bmm_permute.filter")
@registry.reg("rocm.bmm_softmax_bmm_permute_causal.filter")
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
