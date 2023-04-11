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
Grouped back-to-back batched gemm fused kernel, implemented in FMHA style.
Computes bmm(causal_masks(alpha1(activation(alpha0 * bmm(Q, K) [+ bias]))), V),

where:
Q: [B_M0, H, K0] (row_major),
K: [B_N0, H, K0] (column_major),
V: [B_N0, H, N1] (row_major),
bias: [B, H, M0, N0] (row_major). Bias can be omitted.
B_M0, B_N0 are jagged dims.
Layouts are fixed for now.

causal_masks have 3 types:
NO_CAUSAL: no causal masks
UPPER_RIGHT_EMPTY: the upper right triangular part of the matrix is 0
LOWER_LEFT_EMPTY: the bottom left triangular part of the matrix is 0
When causal_masks is enabled, M0 must be equal to N0.

Internally this implementation stores the results of Q@K in shared memory.
It supports larger N0 / N1 compared to the classic_b2b_bmm implementation.
"""

from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.b2b_bmm.fmha_style_b2b_bmm import (
    CausalType,
    fmha_style_b2b_bmm,
)
from aitemplate.utils import shape_utils


class grouped_fmha_style_b2b_bmm(fmha_style_b2b_bmm):
    """See comments at the head of this file."""

    def __init__(
        self,
        causal_type: CausalType,
        epilogue_math_name: str,
        alpha0: float,
        alpha1: float,
        alpha1_divide_by_seq_len: bool = False,
    ) -> None:
        """Initialize grouped_fmha_style_b2b_bmm op.
        Check aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base for more details
        about these args.
        """
        super().__init__(
            causal_type, epilogue_math_name, alpha0, alpha1, alpha1_divide_by_seq_len
        )
        self._attrs["op"] = "grouped_fmha_style_b2b_bmm"

    def _infer_shapes(self):
        """infer the output shape for grouped_fmha_style_b2b_bmm."""
        q, k, v = self._attrs["inputs"][0:3]
        if not (q.is_jagged() and k.is_jagged() and v.is_jagged()):
            raise RuntimeError(f"{q=}, {k=}, {v=} must be jagged!")
        q_shape = q._attrs["shape"]
        k_shape = k._attrs["shape"]
        v_shape = v._attrs["shape"]
        if len(q_shape) != len(k_shape) or len(q_shape) != len(v_shape):
            raise RuntimeError(
                f"QKV ranks must be the same! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        if len(q_shape) != 3:
            raise RuntimeError(
                f"QKV must have rank == 3! Current rank: {len(q_shape)}, QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )

        if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
            raise RuntimeError(
                f"QKV must have same jagged_dim (batch_size and seq_length)! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )

        if len(q_shape[0].jagged_dims()) != 1:
            raise RuntimeError(f"{len(q_shape[0].jagged_dims())=} must be 1!")

        if q_shape[1] != k_shape[1] or q_shape[1] != v_shape[1]:
            raise RuntimeError(
                f"QKV must have same head size! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        K0 = q_shape[2]
        if K0 != k_shape[2]:
            raise RuntimeError(
                f"Q K shapes are not compatible! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )

        num_heads = q_shape[1]
        output_shape = [q_shape[0], num_heads, v_shape[2]]

        if len(self._attrs["inputs"]) == 4:
            batch_size = q_shape[0].batch_dim()
            max_seq_length = q_shape[0].jagged_dims()[0].max_value()
            bias = self._attrs["inputs"][3]
            bias_shape = bias._attrs["shape"]
            bias_expected_shape = [
                batch_size,
                num_heads,
                max_seq_length,
                max_seq_length,
            ]
            bias_max_shape = shape_utils.get_broadcast_max_shape(
                bias_shape, bias_expected_shape
            )
            if len(bias_shape) != 4:
                raise RuntimeError(
                    f"Expected bias rank 4. Current bias rank: {len(bias)}."
                )
            if not bias_max_shape[0]:
                raise RuntimeError(
                    f"bias shape is not compatible with Q K! "
                    f"QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}, "
                    f"bias shapes: {bias_shape=}, {bias_expected_shape=}."
                )
            if bias_shape[-1] != max_seq_length:
                raise RuntimeError(
                    f"Bias last dim is not broadcastable! Expected shape: {max_seq_length}, current bias shape: {bias_shape}"
                )
            # See comments below.
            if not isinstance(q_shape[0].jagged_dims()[0].min_value(), IntImm):
                raise RuntimeError(
                    "Jagged dim' min value must be constant!"
                    f"Current value: {q_shape[0].jagged_dims()=}"
                )
        else:
            # Note: jagged_dims min / max values cannot be IntVar, as AIT lacks the feature to set
            # "attributes" dynamically at runtime in general.
            #
            # Assuming the case: Q @ K @ V, Q / K / V are all dense tensor inputs.
            # As a result, Q / K / V have total_length IntVar to represent the first dimension.
            # Then there are make_jagged() ops which take Q / K / V as well as
            # min_seq_len / max_seq_len IntVars as inputs.
            # At runtime, Q / K / V are inputs passed to AIT runtime. However, since
            # min_seq_len / max_seq_len is not bound to any input dimensions,
            # there are no ways for AIT to infer these values. As a result, AIT compilation would
            # fail.
            #
            # To support min_seq_len / max_seq_len IntVars, there must be a way dynamically set
            # them at runtime.
            #
            # When bias is set, max_seq_len can be inferred from bias input.

            if (not isinstance(q_shape[0].jagged_dims()[0].min_value(), IntImm)) or (
                not isinstance(q_shape[0].jagged_dims()[0].max_value(), IntImm)
            ):
                raise RuntimeError(
                    "Jagged dim' min / max values must be constant!"
                    f"Current value: {q_shape[0].jagged_dims()=}"
                )

        return output_shape
