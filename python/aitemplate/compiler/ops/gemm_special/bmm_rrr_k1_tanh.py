# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] nhwc 3 channel to 8 channel padding
"""
from typing import List

from ...base import IntVar, Tensor
from ..gemm_universal import bmm_rrr

# pylint: disable=C0103,W0221,C0200


class bmm_rrr_k1_tanh(bmm_rrr):
    """[summary]

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self):
        """[summary]"""
        super().__init__()
        self._attrs["op"] = "bmm_rrr_k1_tanh"
        self._attrs["f_ab_alignment"] = True
        self._attrs["has_profiler"] = False

    def _infer_shapes(self, a: Tensor, b: Tensor) -> List[IntVar]:
        """Given input tensors, infers output tensor shapes."""

        a_shapes = a._attrs["shape"]
        if len(a_shapes) != 3:
            raise RuntimeError(
                "bmm operand A should have 3 dimensions! Current shape: {}.".format(
                    a_shapes
                )
            )
        b_shapes = b._attrs["shape"]
        if len(b_shapes) != 3:
            raise RuntimeError(
                "bmm operand B should have 3 dimensions! Current shape: {}.".format(
                    b_shapes
                )
            )
        batch_size_a = a_shapes[0]
        batch_size_b = b_shapes[0]
        if batch_size_a != batch_size_b:
            raise RuntimeError(
                "bmm operand A and B should have same batch_size! "
                "Current shape A: {} shape B: {} .".format(a_shapes, b_shapes)
            )
        assert (
            a_shapes[2] == b_shapes[1]
        ), f"bmm operand A and B should have same K dim! Current shape A: {a_shapes}, shape B: {b_shapes}"
        m_values = a_shapes[1]._attrs["values"]
        # TODO: remove shape check after fixing the kernel
        assert all(
            val % 8 == 0 for val in m_values
        ), f"M should be multiples of 8. M: {a_shapes[1]}"
        n_values = b_shapes[2]._attrs["values"]
        assert all(
            val % 8 == 0 for val in n_values
        ), f"N should be multiples of 8. N: {b_shapes[2]}"
        c_shapes = [batch_size_a, a_shapes[1], b_shapes[2]]
        return c_shapes

    def __call__(self, a: Tensor, b: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [a, b]
        self._set_depth()
        output_shape = self._infer_shapes(a, b)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        """This kernel does not require profiling."""
        return
