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
Operator definition for gemm_rcr_softmax.
"""

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr_softmax(gemm_rcr):
    """gemm_rcr_softmax operator."""

    def __init__(self):
        """Initializes gemm_rcr_softmax."""

        super().__init__()
        self._attrs["op"] = "gemm_rcr_softmax"

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        """Performs sanity checks, offline shape inference and returns an output tensor."""

        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b]

        self._sanity_check(a, b)

        output_shape = self._infer_shapes(a, b)
        self._extract_epilogue_alignment(output_shape)

        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]

        self._set_depth()

        output = Tensor(output_shape, src_ops={self}, dtype=a._attrs["dtype"])
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
