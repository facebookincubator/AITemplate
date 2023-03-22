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
special gemm ops
"""
from aitemplate.compiler.ops.gemm_special.batched_dense_vec_jagged_2d_mul import (
    batched_dense_vec_jagged_2d_mul,
)
from aitemplate.compiler.ops.gemm_special.bmm_rcr_n1 import bmm_rcr_n1
from aitemplate.compiler.ops.gemm_special.bmm_rrr_k1_tanh import bmm_rrr_k1_tanh
from aitemplate.compiler.ops.gemm_special.gemm_rrr_small_nk import gemm_rrr_small_nk


__all__ = [
    "batched_dense_vec_jagged_2d_mul",
    "bmm_rcr_n1",
    "bmm_rrr_k1_tanh",
    "gemm_rrr_small_nk",
]
