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

Special kernel for small K and N
K <= 16, N <= 8
A: [M, K] A can be ND with the first N - 1 dimensions as batch dimensions
B: [K, N]
C: [M, N]
"""

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rrr_small_nk(common.gemm):
    """Special gemm kernel for small K and N (K <= 8, N <= 8)
    A: [M, K]
    B: [K, N]
    C: [M, N]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gemm_rrr_small_nk"
        self._attrs["f_ab_alignment"] = True
        self._attrs["has_profiler"] = False

    @staticmethod
    def is_valid_shape(a: Tensor, b: Tensor):
        valid = len(a.shape()) >= 2 and len(b.shape()) == 2
        for idx in range(2):
            dim = b.shape()[idx]
            if not isinstance(dim, IntImm):
                return False
            if idx == 0:
                # check for K <= 16
                valid &= dim.value() <= 16
            else:
                # check for N <= 8
                valid &= dim.value() <= 8
        return valid

    def _infer_shapes(self, a: Tensor, b: Tensor):
        assert (
            a.shape()[-1] == b.shape()[0]
        ), f"gemm_rrr operand A and B should have the same K dim! A shape: {a.shape()}, B shape: {b.shape()}"

        assert gemm_rrr_small_nk.is_valid_shape(
            a, b
        ), "shape (tensor a:{}, tensor b:{}) not valid for gemm_rrr_small_nk".format(
            a.shape(), b.shape()
        )
        return a._attrs["shape"][:-1] + [b._attrs["shape"][1]]

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        """This kernel does not require profiling"""
        return

    def _extract_dims(self, for_profiling=False):
        A_rank = self._attrs["inputs"][0]._rank()
        # (M, K) * (K, N) = (M, N)
        return {
            "M": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=0, dim_idx=list(range(A_rank - 1))
                ),
                common.DimInfo(
                    common.Source.OUTPUT, tensor_idx=0, dim_idx=list(range(A_rank - 1))
                ),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=A_rank - 1),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=0),
            ],
        }

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        self._attrs["inputs"] = [a, b]
        self._set_depth()
        output_shape = self._infer_shapes(a, b)
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        # self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
