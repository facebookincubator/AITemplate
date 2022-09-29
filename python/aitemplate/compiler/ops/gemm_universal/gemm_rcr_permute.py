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

from aitemplate.testing import detect_target

from ...base import Tensor
from ...tensor_accessor import TensorAccessor
from ..common import reshape

from . import gemm_rcr

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr_permute(gemm_rcr):
    def __init__(self, shape: Tuple[int], layout="20314"):
        super().__init__()
        if layout == "20314":
            self._attrs["op"] = "gemm_rcr_permute"
        elif layout == "m2n3":
            self._attrs["op"] = "gemm_rcr_permute_m2n3"
        else:
            raise NotImplementedError("{} is not implemented!".format(layout))

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
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]

        if self._attrs["layout"] == "Permute5D_20314" or (
            self._attrs["layout"] == "Permute5D_m2n3"
            and detect_target().name() == "rocm"
        ):
            m, n = output_shape
            t1, t2, t3 = self._attrs["shape"]
            output_shape = [t2, m.value() // t1, t3, t1, n.value() // t2 // t3]
            return reshape()(output, output_shape)
        else:
            raise NotImplementedError(
                "{} is not implemented!".format(self._attrs["layout"])
            )
