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
GEMM Specialization: GEMM_RCR(A, B) + Bias
"""
from typing import List

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_rcr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr_bias(gemm_rcr):
    """GEMM Specialization: GEMM_RCR(A, B) + Bias

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(N, K).cuda().half()
        Bias = torch.randn(N).cuda().half()

        y = torch.nn.functional.linear(A, B, bias=Bias)
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias"

    def _sanity_check_extra_inputs(self, output_shape: List):
        bias = self._attrs["inputs"][2]
        bias_shapes = bias._attrs["shape"]
        if len(bias_shapes) != 1:
            raise RuntimeError(
                f"Bias should be 1D vector! Current bias shape: {bias_shapes}"
            )
        bias_shape = bias_shapes[0]
        if not isinstance(bias_shape, IntImm):
            raise RuntimeError(
                f"Bias should be fixed 1D vector! Current bias shape: {bias_shape}"
            )
        if output_shape[-1] != bias_shape:
            raise RuntimeError(
                f"GEMM/Bias shape doesn't match! Gemm shape: {output_shape}, bias shape: {bias_shape}"
            )

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b)
        self._sanity_check_extra_inputs(output_shape)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
