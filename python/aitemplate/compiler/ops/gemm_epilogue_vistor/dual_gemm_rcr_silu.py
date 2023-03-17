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
GEMM Specialization: SILU(GEMM_RCR(A, B)) * GEMM_RCR(A, B1)
"""
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class dual_gemm_rcr_silu(gemm_rcr):
    """GEMM Specialization: SILU(GEMM_RCR(A, B)) * GEMM_RCR(A, B1)

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K)
        W = torch.randn(N, K)
        B = torch.randn(N, K)
        Y1 = torch.nn.functional.linear(A, W)
        Y2 = torch.nn.functional.linear(A, B)
        Y = torch.nn.functional.silu(Y1) * Y2
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "dual_gemm_rcr_silu"
        self._attrs["epilogue2"] = "LeftSiLUAndMul"

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        """Infers output shapes for gemm_rcr_bas.

        Parameters
        ----------
        a : Tensor
            Input tensor A.
        b : Tensor
            Input tensor B.
        bias : Tensor
            Input tensor bias. Must be a 1D vector.

        Returns
        -------
        List[IntVar]
            Output tensor shape.
        """
        return super()._infer_shapes(a, b)

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        if b._attrs["shape"][-2] != 1 and bias._attrs["shape"][-2] == 1:
            self._attrs["broadcast_b1"] = True
        return output
