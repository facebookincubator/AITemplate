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

from aitemplate.backend.cuda.gemm_epilogue_vistor import (
    bmm_rcr_softmax,
    dual_bmm_rrr_div,
    dual_gemm_rcr_fast_gelu,
    dual_gemm_rcr_silu,
    gemm_rcr_bias_softmax,
    gemm_rcr_softmax,
)

__all__ = [
    "bmm_rcr_softmax",
    "dual_bmm_rrr_div",
    "dual_gemm_rcr_fast_gelu",
    "dual_gemm_rcr_silu",
    "gemm_rcr_bias_softmax",
    "gemm_rcr_softmax",
]
