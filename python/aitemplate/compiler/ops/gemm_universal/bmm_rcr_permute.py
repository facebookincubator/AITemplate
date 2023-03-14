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
Batch GEMM specialization for A[RowMajor], B[ColMajor], C[RowMajor] with permutation on output.
"""

from typing import Tuple

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common import reshape
from aitemplate.compiler.ops.gemm_universal.bmm_xxx import bmm_rcr
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class bmm_rcr_permute(bmm_rcr):
    """Batch GEMM specialization for A[RowMajor], B[ColMajor], C[RowMajor] with permutation
    on output to given layout.

    Currently only supports reshape to 4D tensor, then do 0213 permute

    This operator is equivalent to following PyTorch code:

    .. highlight:: python
    .. code-block:: python
        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        WT = torch.transpose(W_pt, 2, 1)
        Y_l = torch.bmm(X_pt, WT)
        Y_r = Y_l.reshape(B // D1, D1, M, N)
        Y_pt = torch.permute(Y_r, [0, 2, 1, 3])
    """

    def __init__(self, shape: Tuple[int], layout="0213"):
        """Constructor for bmm_rcr_permute

        Parameters
        ----------
        shape : Tuple[int]
            Necessary dim info of the reshape operator
            In 0213 case, we need to know the [D1,] to reshape the output from 3D to 4D
        layout : str, optional
            permutation type, by default "0213"
        """
        super().__init__()
        self._attrs["op"] = "bmm_rcr_permute"
        self._attrs["shape"] = shape
        self._attrs["layout"] = "Permute4DBMM_{}".format(layout)
        self._attrs["permute_shape"] = "_".join(map(str, shape))

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        """Call bmm_rcr_permute with tensors a, b

        Parameters
        ----------
        a : Tensor
            Tensor in shape (B, M, K)
        b : Tensor
            Tensor in shape (B, N, K)

        Returns
        -------
        Tensor
            Tensors in shape (B // D1, M, D1, N) for 0213 permute

        Raises
        ------
        NotImplementedError
            Permute layout not implemented yet
        """
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b]
        self._attrs["input_accessors"] = [TensorAccessor(a), TensorAccessor(b)]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b)
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]

        if self._attrs["layout"] == "Permute4DBMM_0213":
            b, m, n = output_shape
            d1 = self._attrs["shape"][0]
            output_shape = [b.value() // d1, m, d1, n]
            self._extract_epilogue_alignment(output_shape)
            return reshape()(output, output_shape)
        else:
            raise NotImplementedError(
                "{} is not implemented!".format(self._attrs["layout"])
            )

    def _get_op_attributes(self):
        return {
            "layout": self._attrs["layout"].split("_")[-1],
            "shape": tuple(map(int, self._attrs["permute_shape"].split("_"))),
        }
