# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
gemm rrr with bias + permute
"""

from typing import Tuple

from aitemplate.testing import detect_target

from ...base import Tensor
from ...tensor_accessor import TensorAccessor
from ..common import reshape

from . import gemm_rrr_bias

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rrr_bias_permute(gemm_rrr_bias):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self, shape: Tuple[int], layout="20314"):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "gemm_rrr_bias_permute"
        self._attrs["shape"] = shape
        self._attrs["layout"] = "Permute5D_{}".format(layout)

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]

        if self._attrs["layout"] == "Permute5D_20314":
            m, n = output_shape
            t1, t2, t3 = self._attrs["shape"]
            # TODO:currently ROCM only partitions M by 2 time and N by 1 time.
            # We should update ROCM to use the same output_shape
            if detect_target().name() == "rocm":
                output_shape = [t2, m.value() // t1 // t2, t3, t1, n.value() // t3]
            else:
                output_shape = [t2, m.value() // t1, t3, t1, n.value() // t3 // t2]
            return reshape()(output, output_shape)
        else:
            raise NotImplementedError(
                "{} is not implemented!".format(self._attrs["layout"])
            )
