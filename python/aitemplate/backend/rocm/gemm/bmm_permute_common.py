# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Common template for bmm
"""
import jinja2

EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/device_batched_gemm_e_permute_xdl.hpp"
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{{indent}}                                M * K, // batch_stride_A
{{indent}}                                N * K, // batch_stride_B
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% endif %}
{{indent}}                                {int(B/G1), G1, int(M), int(N), int(M*G1*N), int(N), int(G1*N), 1},
{{indent}}                                B,
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "bias" %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% endif %}
"""
)

EXTRA_PERM_ARGS_TEMPLATE = jinja2.Template(
    """
const int G1={{g1}};
"""
)
