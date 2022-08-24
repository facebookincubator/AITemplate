# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When used for `linear`, need to set A->Data, B->Weight
"""

from aitemplate.compiler.tensor_accessor import TensorAccessor

from ...base import Tensor
from . import bmm_rrr

# pylint: disable=C0103, W0223


class bmm_rrr_add(bmm_rrr):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "bmm_rrr_add"
        self._attrs["has_d"] = True

    def __call__(self, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        output = super().__call__(a, b)
        self._attrs["inputs"].append(c)
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        return output
