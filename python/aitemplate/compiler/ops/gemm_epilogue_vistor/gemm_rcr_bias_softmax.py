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
Operator definition for gemm_rcr_bias_softmax.
"""
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_epilogue_vistor.gemm_rcr_softmax import (
    gemm_rcr_softmax,
)
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,R1711,W0102,W0221,E1120,W0223


class gemm_rcr_bias_softmax(gemm_rcr_softmax):
    """gemm_rcr_bias_softmax operator."""

    def __init__(self):
        """Initializes gemm_rcr_bias_softmax."""

        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias_softmax"

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        """Infers output shapes from input tensors."""

        bias_shape = bias._attrs["shape"]
        if len(bias_shape) != 1:
            raise RuntimeError("Bias should be 1D vector ")
        bias_shape_value = bias_shape[0]._attrs["values"]
        if len(bias_shape_value) != 1:
            raise RuntimeError("Bias should be fixed 1D vector")
        bias_dim = bias_shape_value[0]
        outshape = super()._infer_shapes(a, b)
        if outshape[1]._attrs["values"][0] != bias_dim:
            raise RuntimeError("GEMM/Bias shape doesn't match")
        return outshape

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        """Performs sanity checks, offline shape inference and returns an output tensor."""

        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias]

        self._sanity_check(a, b)

        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)

        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]

        self._set_depth()

        output = Tensor(output_shape, src_ops={self}, dtype=a._attrs["dtype"])
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
