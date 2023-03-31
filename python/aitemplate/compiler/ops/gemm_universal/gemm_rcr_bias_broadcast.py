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

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_rcr_bias
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103, W0223, W0221


class gemm_rcr_bias_broadcast(gemm_rcr_bias):
    def __init__(self):
        super().__init__()
        self._attrs["epilogue"] = "LinearCombinationResidualBlock"

    @staticmethod
    def is_valid_inputs(*inputs):
        msg = ""
        if len(inputs) < 3:
            msg = "input for gemm_rcr_bias_broadcast should be at least 3, got {} instead.".format(
                len(inputs)
            )
            return False, msg

        gemm_rcr_bias_valid, msg = gemm_rcr_bias.is_valid_inputs(
            inputs[0], inputs[1], inputs[2]
        )
        if not gemm_rcr_bias_valid:
            return False, msg

        if len(inputs) > 3:
            base_shape = gemm_rcr_bias()._infer_shapes(inputs[0], inputs[1], inputs[2])
            for d in inputs[3:]:
                d_shape = d.shape()
                if d_shape != base_shape:
                    msg = f"Additional elementwise shape {d_shape} doesn't match gemm_bias' shape {base_shape}"
                    return False, msg

        return True, msg

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
        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
