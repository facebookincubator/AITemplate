# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row))
in torch it is
# _2905_2929 = _2904.view(B, 25, -1).permute(1, 0, 2)
# _2930_2954 = torch.baddbmm(
#      self._1085_1133, _2905_2929, self._1084_1132) # baddbmm(bias, X, W)
"""

from ...base import IntImm, Tensor
from ...tensor_accessor import TensorAccessor
from . import perm102_bmm_rrr

# pylint: disable=C0103, W0223, W0221


class perm102_bmm_rrr_bias(perm102_bmm_rrr):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "perm102_bmm_rrr_bias"

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        bias_shapes = bias._attrs["shape"]
        if len(bias_shapes) != 2:
            raise RuntimeError("Bias should be 2D vector ")
        bias_shape = bias_shapes[1]
        if not isinstance(bias_shape, IntImm):
            raise RuntimeError("Bias should be fixed 2D vector")
        outshape = super()._infer_shapes(a, b)
        if outshape[2] != bias_shape:
            raise RuntimeError("GEMM/Bias shape doesn't match")
        return outshape

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        """_summary_

        Parameters
        ----------
        a : Tensor
            _description_
        b : Tensor
            _description_
        bias : Tensor
            _description_

        Returns
        -------
        Tensor
            _description_
        """
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
        return output
