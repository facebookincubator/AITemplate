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
GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""

from typing import Tuple

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common import reshape

from aitemplate.compiler.ops.gemm_universal import gemm_rrr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rrr_permute(gemm_rrr):
    def __init__(self, shape: Tuple[int], layout="20314"):
        super().__init__()
        self._attrs["op"] = "gemm_rrr_permute"
        self._attrs["shape"] = shape
        self._attrs["layout"] = "Permute5D_{}".format(layout)
        self._attrs["permute_shape"] = "_".join(map(str, shape))

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b]
        self._attrs["input_accessors"] = [TensorAccessor(a), TensorAccessor(b)]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b)

        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]

        if self._attrs["layout"] == "Permute5D_20314":
            m, n = output_shape
            t1, t2, t3 = self._attrs["shape"]
            output_shape = [t2, m.value() // t1, t3, t1, n.value() // t2 // t3]
            self._extract_epilogue_alignment(output_shape)
            return reshape()(output, output_shape)
        else:
            raise NotImplementedError(
                "{} is not implemented!".format(self._attrs["layout"])
            )

    def _get_op_attributes(self):
        return {
            "layout": self._attrs["layout"].split("_")[-1],
            "shape": self._attrs["shape"],
        }
