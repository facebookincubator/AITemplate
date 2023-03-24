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
gemm_rcr_bias with 2 extra sources.
BinaryOp2(BinaryOp1(UnaryOp(TensorOp(X) + bias), residual1), residual2)
"""
from typing import List

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_rcr_bias
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103, W0223, W0221


class gemm_rcr_bias_broadcast(gemm_rcr_bias):
    def __init__(self):
        super().__init__()
        self._attrs["epilogue"] = "LinearCombinationResidualBlock"

    def _sanity_check_extra_inputs(self, output_shape: List):
        super()._sanity_check_extra_inputs(output_shape)
        inputs = self._attrs["inputs"]
        for d in inputs[3:]:
            d_shape = d.shape()
            if d_shape != output_shape:
                raise RuntimeError(
                    f"Additional elementwise shape {d_shape} doesn't match gemm_rcr_bias' shape {output_shape}"
                )

    def __call__(
        self, a: Tensor, b: Tensor, bias: Tensor, d0: Tensor, d1: Tensor = None
    ) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias, d0]
        if d1 is not None:
            self._attrs["inputs"].append(d1)
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
