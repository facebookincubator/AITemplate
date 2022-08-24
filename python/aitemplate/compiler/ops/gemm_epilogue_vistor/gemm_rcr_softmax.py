# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""

from ...base import IntImm, Tensor
from ...tensor_accessor import TensorAccessor
from ..gemm_universal.gemm_rcr import gemm_rcr

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

        temp_c = Tensor(output_shape, dst_ops={self})
        temp_d = Tensor(output_shape, dst_ops={self})
        temp_n = Tensor([output_shape[0], IntImm(1)], dtype="float32", dst_ops={self})

        self._attrs["inputs"].append(temp_c)
        self._attrs["inputs"].append(temp_d)
        self._attrs["inputs"].append(temp_n)
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]

        self._set_depth()

        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
