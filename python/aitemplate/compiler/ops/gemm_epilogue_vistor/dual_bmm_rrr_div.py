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
Batch GEMM specialization: BMM_RRR(A, B0) / BMM_RRR(A, B1)
"""
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_universal import bmm_rrr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class dual_bmm_rrr_div(bmm_rrr):
    """Batch GEMM specialization: BMM_RRR(A, B0) / BMM_RRR(A, B1)

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(B, M, K)
        B0 = torch.randn(B, K, N)
        B1 = torch.randn(B, K, N)
        D0 = torch.bmm(A, B0)
        D1 = torch.bmm(A, B1)
        D2 = D0 / D1

    If the last dim of B1 is 1 (while the last dim of B0 isn't),
    B1 is broadcasted to the same shape as B0 before computing
    the right gemm A @ B1.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "dual_bmm_rrr_div"
        self._attrs["epilogue2"] = "Div"

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        output = super().__call__(a, b)
        self._attrs["inputs"].append(bias)
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        if b._attrs["shape"][-1] != 1 and bias._attrs["shape"][-1] == 1:
            self._attrs["broadcast_b1"] = True
        return output
