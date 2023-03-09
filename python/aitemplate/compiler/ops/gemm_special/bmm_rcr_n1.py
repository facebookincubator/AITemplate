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
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight

Special kernel for GEMV case:
A: [B, M, K]
B: [B, N, K]
C: [B, M, N]
where N = 1

This kernel computes C = alpha * A @ B
"""

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import bmm_rcr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103, W0223, W0221, W0613


class bmm_rcr_n1(bmm_rcr):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "bmm_rcr_n1"
        self._attrs["f_ab_alignment"] = True
        self._attrs["has_profiler"] = False

    @staticmethod
    def is_valid_shape(a: Tensor, b: Tensor):
        """
        Check input a/b is valid for bmm_rcr_n1.
        Requirements:
            1) matching dimension of a/b (where a is row major, b is column major)
            2) dim N of b needs to be 1
            3) dim K of b needs to be multiple of 8
        """
        if len(a.shape()) != 3 or len(b.shape()) != 3:
            return False

        valid = True
        valid &= a.shape()[0] == b.shape()[0]
        valid &= a.shape()[2] == b.shape()[2]

        # check N = 1
        BN = b.shape()[1]
        if not isinstance(BN, IntImm):
            return False
        valid &= BN.value() == 1

        # check BK is static dim
        BK = b.shape()[2]
        if not isinstance(BK, IntImm):
            return False

        return valid

    def _infer_shapes(self, a: Tensor, b: Tensor):
        assert self.is_valid_shape(
            a, b
        ), "shape (tensor a:{}, tensor b:{}) not valid for bmm_rcr_n1".format(
            a.shape(), b.shape()
        )
        return super()._infer_shapes(a, b)

    def __call__(self, a: Tensor, b: Tensor, alpha: float = 1.0) -> Tensor:
        self._attrs["inputs"] = [a, b]
        self._attrs["alpha"] = alpha
        self._set_depth()
        output_shape = self._infer_shapes(a, b)
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["input_accessors"] = [TensorAccessor(a), TensorAccessor(b)]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        """This kernel doesn't require profiling."""
        return
