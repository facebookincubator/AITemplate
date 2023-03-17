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
gemm rrr with bias
"""
from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_rrr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rrr_bias(gemm_rrr):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gemm_rrr_bias"

    @staticmethod
    def is_valid_inputs(X: Tensor, W: Tensor, bias: Tensor):
        msg = ""

        bias_shapes = bias._attrs["shape"]
        if len(bias_shapes) != 1:
            msg = f"Bias should be 1D vector! Current bias shape: {bias_shapes}"
            return False, msg

        bias_shape = bias_shapes[0]
        if not isinstance(bias_shape, IntImm):
            msg = f"Bias should be fixed 1D vector! Current bias shape: {bias_shape}"
            return False, msg

        outshape = gemm_rrr()._infer_shapes(X, W)
        if outshape[-1] != bias_shape:
            msg = f"GEMM/Bias shape doesn't match! Gemm shape: {outshape}, bias shape: {bias_shape}"
            return False, msg

        return True, msg

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        """Infers output shapes for gemm_rrr_bas.

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
        is_valid_inputs, msg = self.is_valid_inputs(a, b, bias)
        if not is_valid_inputs:
            raise RuntimeError(msg)
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
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
