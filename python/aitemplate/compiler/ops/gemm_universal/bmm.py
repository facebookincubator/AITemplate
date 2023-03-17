#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Base class for Batch GEMM.
"""

# pylint: disable=C0103,W0223

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.dtype import is_same_dtype
from aitemplate.compiler.ops.gemm_universal import gemm_common as common
from aitemplate.compiler.ops.gemm_universal.gemm_common import gemm


def is_valid_inputs(output_shapes, c_shapes):
    """
    Used by bmm_xxx_add ops to check whether elementwise ops
    can be fused to the bmm op via epilogue fusion. So far,
    only add ops are supported.
    """
    msg = ""
    if output_shapes == c_shapes:
        return True, msg

    def _squeeze_leading_1s(shapes):
        out = []
        if len(shapes) == 0:
            return out
        i = 0
        for shape in shapes:
            if not isinstance(shape, IntImm):
                break
            if shape.value() != 1:
                break
            i = i + 1

        out = shapes[i:]
        if len(out) == 0:
            out.append(shapes[-1])
        return out

    msg = (
        f"C can't be broadcast to the bmm output."
        f"Output shapes: {output_shapes}, C shapes: {c_shapes}"
    )
    bias_shapes = _squeeze_leading_1s(c_shapes)
    if len(bias_shapes) >= len(output_shapes):
        return False, msg

    for o_shape, c_shape in zip(reversed(output_shapes), reversed(bias_shapes)):
        if o_shape != c_shape:
            return False, msg

    return True, ""


class bmm(gemm):
    """Base class for bmm."""

    def _get_batch_size(self, a: Tensor, b: Tensor):
        self._sanity_check(a, b)

        a_shapes = a._attrs["shape"]
        b_shapes = b._attrs["shape"]
        if len(a_shapes) == 2:
            return b_shapes[0]
        elif len(b_shapes) == 2:
            return a_shapes[0]

        batch_size_a = a_shapes[0]
        batch_size_b = b_shapes[0]
        if batch_size_a != batch_size_b and batch_size_a != 1 and batch_size_b != 1:
            raise RuntimeError(
                "bmm operand A and B should have same batch_size, or batch_size = 1! "
                "Current shape A: {} shape B: {} .".format(a_shapes, b_shapes)
            )

        return a_shapes[0] if a_shapes[0] != 1 else b_shapes[0]

    def _sanity_check(self, a: Tensor, b: Tensor):
        a_shapes = a._attrs["shape"]
        if len(a_shapes) != 2 and len(a_shapes) != 3:
            raise RuntimeError(
                "bmm operand A should have 2 or 3 dimensions! Current shape: {}.".format(
                    a_shapes
                )
            )
        b_shapes = b._attrs["shape"]
        if len(b_shapes) != 2 and len(b_shapes) != 3:
            raise RuntimeError(
                "bmm operand B should have 2 or 3 dimensions! Current shape: {}.".format(
                    b_shapes
                )
            )
        if len(a_shapes) == 2 and len(b_shapes) == 2:
            raise RuntimeError(
                "bmm operand A and B both have 2 dimensions! Use gemm instead."
            )
        if not is_same_dtype(a.dtype(), b.dtype()):
            raise RuntimeError(
                "gemm operand A and B should have the same data type! Current A: {atype}, B: {btype}.".format(
                    atype=a.dtype(), btype=b.dtype()
                )
            )

    def _invert_exec_key(self, key):
        return common.gemm_inverse_key_func(key)
