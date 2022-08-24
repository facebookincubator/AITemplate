# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""

from typing import Tuple

from ...base import Tensor
from ...tensor_accessor import TensorAccessor
from ..common import reshape
from . import bmm_rcr

# pylint: disable=C0103,W0223,W0221,W0613


class bmm_rcr_permute(bmm_rcr):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self, shape: Tuple[int], layout="0213"):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "bmm_rcr_permute"
        self._attrs["shape"] = shape
        self._attrs["layout"] = "Permute4DBMM_{}".format(layout)

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

        if self._attrs["layout"] == "Permute4DBMM_0213":
            b, m, n = output_shape
            d1 = self._attrs["shape"][0]
            output_shape = [b.value() // d1, m, d1, n]
            return reshape()(output, output_shape)
        else:
            raise NotImplementedError(
                "{} is not implemented!".format(self._attrs["layout"])
            )
