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
Common template for bmm
"""
import jinja2

EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_e_permute_xdl.hpp"
"""
)

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}const int64_t G1 = p_dim0; // G1

{{indent}}const int64_t stride_a = *a_dim2;
{{indent}}const int64_t stride_b = *b_dim2;
{{indent}}const int64_t stride_c = *c_dim2;
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
{{indent}}                                {int(B/G1), int(G1), int(M), int(N), int(M*G1*N), int(N), int(G1*N), 1},
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
