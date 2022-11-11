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
Batch GEMM specialization: C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row)) + bias[b, n]
"""

from ...base import IntImm, Tensor
from ...tensor_accessor import TensorAccessor
from . import perm102_bmm_rrr

# pylint: disable=C0103, W0223, W0221


class perm102_bmm_rrr_bias(perm102_bmm_rrr):
    """Batch GEMM specialization: C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row)) + bias[b, n]

    The op is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python
        X_pt = torch.randn(M, B, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()
        B_pt = torch.randn(B, N).cuda().half()

        XT = X_pt.permute(1, 0, 2)
        Bias = B_pt.unsqueeze(1)
        Y_pt = torch.baddbmm(Bias, XT, W_pt)
        Y_pt = Y_pt.permute(1, 0, 2)
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "perm102_bmm_rrr_bias"

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        bias_shapes = bias._attrs["shape"]
        if len(bias_shapes) != 2:
            raise RuntimeError("Bias should be 2D vector ")
        bias_shape = bias_shapes[1]
        if not isinstance(bias_shape, IntImm):
            raise RuntimeError("Bias should be fixed 2D vector")
        outshape = super()._infer_shapes(a, b)
        if outshape[2] != bias_shape:
            raise RuntimeError("GEMM/Bias shape doesn't match")
        return outshape

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        self._attrs["inputs"] = [a, b, bias]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
