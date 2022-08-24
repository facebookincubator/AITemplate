# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ...base import Tensor
from ...tensor_accessor import TensorAccessor
from .gemm_rcr_softmax import gemm_rcr_softmax

# pylint: disable=C0103,R1711,W0102,W0221,E1120,W0223


class gemm_rcr_bias_softmax(gemm_rcr_softmax):
    """gemm_rcr_bias_softmax operator."""

    def __init__(self):
        """Initializes gemm_rcr_bias_softmax."""

        super().__init__()
        self._attrs["op"] = "gemm_rcr_bias_softmax"

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        """Infers output shapes from input tensors."""

        bias_shape = bias._attrs["shape"]
        if len(bias_shape) != 1:
            raise RuntimeError("Bias should be 1D vector ")
        bias_shape_value = bias_shape[0]._attrs["values"]
        if len(bias_shape_value) != 1:
            raise RuntimeError("Bias should be fixed 1D vector")
        bias_dim = bias_shape_value[0]
        outshape = super()._infer_shapes(a, b)
        if outshape[1]._attrs["values"][0] != bias_dim:
            raise RuntimeError("GEMM/Bias shape doesn't match")
        return outshape

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        """Performs sanity checks, offline shape inference and returns an output tensor."""

        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, bias]

        self._sanity_check(a, b)

        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)

        temp_d = Tensor(output_shape, dst_ops={self})
        temp_n = Tensor([output_shape[0], 1], dtype="float32", dst_ops={self})

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
