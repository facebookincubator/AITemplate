# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from ...base import Tensor
from ...tensor_accessor import TensorAccessor
from ..common import reshape
from .perm021fc_ccr_bias import perm021fc_ccr_bias

# pylint: disable=C0103,W0223,W0221,W0613


class perm021fc_ccr_bias_permute(perm021fc_ccr_bias):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self, layout="021"):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "perm021fc_ccr_bias_permute"
        self._attrs["shape"] = [0]  # this is a hack
        self._attrs["layout"] = "Permute3DBMM_{}".format(layout)

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias]
        self._attrs["input_accessors"] = [TensorAccessor(a), TensorAccessor(b)]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)

        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]

        if self._attrs["layout"] == "Permute3DBMM_021":
            b, m, n = output_shape
            output_shape = [b, n, m]
            self._attrs["epilogue_alignment"] = 1
            return reshape()(output, output_shape)
        else:
            raise NotImplementedError(
                "{} is not implemented!".format(self._attrs["layout"])
            )
