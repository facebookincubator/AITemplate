# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[b, m, n] = bmm([b, k, m], [1, n, k])
in torch it is
...
# _3306 = _3305.permute(0, 2, 1)  # Transpose
# _3307 = _3306  # torch.reshape(_3306, (-1, 745))  # Reshape
# _3308 = torch.nn.functional.linear(_3307, self._1184, bias=self._1185)  # FC
"""

from aitemplate.compiler.tensor_accessor import TensorAccessor

from ...base import Tensor
from . import perm021fc_ccr

# pylint: disable=C0103, W0223, W0221


class perm021fc_ccr_bias(perm021fc_ccr):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "perm021fc_ccr_bias"

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        """_summary_

        Parameters
        ----------
        a : Tensor
            _description_
        b : Tensor
            _description_
        """
        bias_shape = bias._attrs["shape"]
        if len(bias_shape) != 1:
            raise RuntimeError("Bias should be 1D vector ")
        bias_shape_value = bias_shape[0]._attrs["values"]
        if len(bias_shape_value) != 1:
            raise RuntimeError("Bias should be fixed 1D vector")
        bias_dim = bias_shape_value[0]
        outshape = super()._infer_shapes(a, b)
        if outshape[2]._attrs["values"][0] != bias_dim:
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
        self._attrs["output_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["outputs"]
        ]
        return output
