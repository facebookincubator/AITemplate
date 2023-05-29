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
Rocm gemm init.
"""
from aitemplate.backend.rocm.gemm import (  # noqa: F401
    bmm_ccr,
    bmm_ccr_add,
    bmm_crr,
    bmm_crr_add,
    bmm_rcr,
    bmm_rcr_permute,
    bmm_rrr,
    bmm_rrr_add,
    bmm_rrr_permute,
    bmm_softmax_bmm,
    bmm_softmax_bmm_permute,
    gemm_rcr,
    gemm_rcr_bias,
    gemm_rcr_bias_add,
    gemm_rcr_bias_add_add,
    gemm_rcr_bias_add_add_relu,
    gemm_rcr_bias_add_relu,
    gemm_rcr_bias_fast_gelu,
    gemm_rcr_bias_mul,
    gemm_rcr_bias_mul_add,
    gemm_rcr_bias_mul_tanh,
    gemm_rcr_bias_permute,
    gemm_rcr_bias_permute_m2n3,
    gemm_rcr_bias_permute_m3n2,
    gemm_rcr_bias_relu,
    gemm_rcr_bias_sigmoid,
    gemm_rcr_bias_sigmoid_mul,
    gemm_rcr_bias_sigmoid_mul_tanh,
    gemm_rcr_bias_swish,
    gemm_rcr_bias_tanh,
    gemm_rcr_permute_m2n3,
    gemm_rrr,
    gemm_rrr_bias_permute,
)
