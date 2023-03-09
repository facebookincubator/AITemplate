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
"""Grouped GEMM Specialization: ReLU(GEMM_RCR(A, B) + Bias)
"""

from aitemplate.compiler.ops.gemm_universal import group_gemm_rcr_bias

# pylint: disable=C0103,W0223


class group_gemm_rcr_bias_relu(group_gemm_rcr_bias):
    """Grouped GEMM Specialization: ReLU(GEMM_RCR(A, B) + Bias)

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        # group 1
        A1 = torch.randn(M1, K1).cuda().half()
        B1 = torch.randn(N1, K1).cuda().half()
        Bias1 = torch.randn(N1).cuda().half()

        linear1 = torch.nn.functional.linear(A1, B1, bias=Bias1)
        y1 = torch.nn.ReLU(linear1)

        ...
        # group n
        An = torch.randn(Mn, Kn).cuda().half()
        Bn = torch.randn(Nn, Kn).cuda().half()
        Biasn = torch.randn(Nn).cuda().half()

        linearn = torch.nn.functional.linear(An, Bn, bias=Biasn)
        yn = torch.nn.ReLU(linearn)
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "group_gemm_rcr_bias_relu"
        self._attrs["epilogue"] = "LinearCombinationRelu"
