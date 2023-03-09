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
gemm rcr with bias + permute
"""

from typing import Tuple

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common import reshape
from aitemplate.compiler.ops.gemm_universal import gemm_rcr_bias
from aitemplate.compiler.tensor_accessor import TensorAccessor

from aitemplate.testing import detect_target

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr_bias_permute(gemm_rcr_bias):
    def __init__(self, shape: Tuple[int], layout="20314"):
        super().__init__()
        if layout == "20314":
            self._attrs["op"] = "gemm_rcr_bias_permute"
        elif layout == "m2n3":
            self._attrs["op"] = "gemm_rcr_bias_permute_m2n3"
        elif layout == "m3n2":
            self._attrs["op"] = "gemm_rcr_bias_permute_m3n2"
        else:
            raise NotImplementedError("{} is not implemented!".format(layout))
        self._attrs["shape"] = shape
        self._attrs["layout"] = "Permute5D_{}".format(layout)
        self._attrs["permute_shape"] = "_".join(map(str, shape))

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, bias)

        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]

        m, n = output_shape
        t1, t2, t3 = self._attrs["shape"]
        if (
            self._attrs["layout"] == "Permute5D_20314"
            and detect_target().name() == "rocm"
        ) or self._attrs["layout"] == "Permute5D_m3n2":
            output_shape = [t2, m.value() // t1 // t2, t3, t1, n.value() // t3]
        else:
            output_shape = [t2, m.value() // t1, t3, t1, n.value() // t3 // t2]
        self._extract_epilogue_alignment(output_shape)
        return reshape()(output, output_shape)

    def _get_op_attributes(self):
        return {
            "layout": self._attrs["layout"].split("_")[-1],
            "shape": self._attrs["shape"],
        }
