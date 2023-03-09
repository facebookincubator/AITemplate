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
GEMM Specialization: FastGELU(GEMM_RCR(A, B) + Bias)
"""
from aitemplate.compiler.ops.gemm_universal import gemm_rcr_bias

# pylint: disable=C0103,W0223,W0221


class gemm_rcr_bias_fast_gelu(gemm_rcr_bias):
    """GEMM Specialization: FastGELU(GEMM_RCR(A, B) + Bias)

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(N, K).cuda().half()
        Bias = torch.randn(N).cuda().half()

        linear = torch.nn.functional.linear(A, B, bias=Bias)
        y = torch.nn.GELU(linear)
    """

    def __init__(self):
        """Constructor for gemm_rcr_bias_fast_gelu"""
        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias_fast_gelu"
        self._attrs["epilogue"] = "LinearCombinationFastGELU"
