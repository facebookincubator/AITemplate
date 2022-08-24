# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
gemm rcr with bias
"""
from ...base import IntImm, Tensor
from ...tensor_accessor import TensorAccessor
from . import gemm_rcr

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr_bias(gemm_rcr):
    """_summary_

    Parameters
    ----------
    gemm_rcr : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias"

    @staticmethod
    def is_valid_inputs(X: Tensor, W: Tensor, bias: Tensor):
        msg = ""

        bias_shapes = bias._attrs["shape"]
        if len(bias_shapes) != 1:
            msg = f"Bias should be 1D vector! Current bias shape: {bias_shapes}"
            return False, msg

        bias_shape = bias_shapes[0]
        if not isinstance(bias_shape, IntImm):
            msg = f"Bias should be fixed 1D vector! Current bias shape: {bias_shape}"
            return False, msg

        outshape = gemm_rcr()._infer_shapes(X, W)
        if outshape[-1] != bias_shape:
            msg = f"GEMM/Bias shape doesn't match! Gemm shape: {outshape}, bias shape: {bias_shape}"
            return False, msg

        return True, msg

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        """Infers output shapes for gemm_rcr_bas.

        Parameters
        ----------
        a : Tensor
            Input tensor A.
        b : Tensor
            Input tensor B.
        bias : Tensor
            Input tensor bias. Must be a 1D vector.

        Returns
        -------
        List[IntVar]
            Output tensor shape.
        """
        is_valid_inputs, msg = self.is_valid_inputs(a, b, bias)
        if not is_valid_inputs:
            raise RuntimeError(msg)
        return super()._infer_shapes(a, b)

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
        return output
