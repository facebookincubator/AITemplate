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
Back-to-back batched gemm fused kernel, implemented in FMHA style.
Computes bmm(causal_masks(alpha1(activation(alpha0 * bmm(Q, K) [+ bias]))), V),

where:
Q: [B, M0, H, K0] (row_major),
K: [B, N0, H, K0] (column_major),
V: [B, N0, H, N1] (row_major),
bias: [B, H, M0, N0] (row_major). Bias can be omitted.
Layouts are fixed for now.

causal_masks have 3 types:
NO_CAUSAL: no causal masks
UPPER_RIGHT_EMPTY: the upper right triangular part of the matrix is 0
LOWER_LEFT_EMPTY: the bottom left triangular part of the matrix is 0
When causal_masks is enabled, M0 must be equal to N0.

Internally this implementation stores the results of Q@K in shared memory.
It supports larger N0 / N1 compared to the classic_b2b_bmm implementation.
"""

from typing import Optional

import numpy as np

from aitemplate.backend import registry, target
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import b2b_bmm_base, CausalType
from aitemplate.utils import shape_utils


class fmha_style_b2b_bmm(b2b_bmm_base):
    """See comments at the head of this file."""

    def __init__(
        self,
        causal_type: CausalType,
        epilogue_math_name: str,
        alpha0: float,
        alpha1: float,
        alpha1_divide_by_seq_len: bool = False,
    ) -> None:
        """Initialize fmha_style_b2b_bmm op.
        Check aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base for more details
        about these args.
        """
        super().__init__(
            causal_type, epilogue_math_name, alpha0, alpha1, alpha1_divide_by_seq_len
        )
        self._attrs["op"] = "fmha_style_b2b_bmm"
        self._attrs["workspace"] = 0

    def _infer_shapes(self):
        """infer the output shape for fmha_style_b2b_bmm."""
        q, k, v = self._attrs["inputs"][0:3]
        q_shape = q._attrs["shape"]
        k_shape = k._attrs["shape"]
        v_shape = v._attrs["shape"]
        if len(q_shape) != len(k_shape) or len(q_shape) != len(v_shape):
            raise RuntimeError(
                f"QKV ranks must be the same! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        if len(q_shape) != 4:
            raise RuntimeError(
                f"QKV must have rank == 4! Current rank: {len(q_shape)}, QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )

        if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
            raise RuntimeError(
                f"QKV must have same batch size! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        if q_shape[2] != k_shape[2] or q_shape[2] != v_shape[2]:
            raise RuntimeError(
                f"QKV must have same head size! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        batch_size = q_shape[0]
        M0 = q_shape[1]
        K0 = q_shape[3]
        if K0 != k_shape[3]:
            raise RuntimeError(
                f"Q K shapes are not compatible! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        N0 = k_shape[1]
        if N0 != v_shape[1]:
            raise RuntimeError(
                f"K V shapes are not compatible! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        N1 = v_shape[3]

        if self._attrs["causal_type"] != CausalType.NO_CAUSAL:
            if M0 != N0:
                raise RuntimeError(
                    f"When causal_type is enabled, M0 must be equal to N0. Current {M0=}, {N0=}."
                )

        head_size = q_shape[2]
        output_shape = [batch_size, M0, head_size, N1]

        if len(self._attrs["inputs"]) == 4:
            bias = self._attrs["inputs"][3]
            bias_shape = bias._attrs["shape"]
            bias_expected_shape = [batch_size, head_size, M0, N0]
            broadcastable, _ = shape_utils.get_broadcast_max_shape(
                bias_shape, bias_expected_shape
            )
            if len(bias_shape) != 4:
                raise RuntimeError(
                    f"Expected bias rank 4. Current bias rank: {len(bias_shape)}."
                )
            if not broadcastable:
                raise RuntimeError(
                    f"bias shape is not compatible with Q K! "
                    f"QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}, "
                    f"bias shapes: {bias_shape=}, {bias_expected_shape=}."
                )
            if bias_shape[-1] != N0:
                raise RuntimeError(
                    f"Bias last dim is not broadcastable! Expected shape: {N0}, current bias shape: {bias_shape}"
                )
        return output_shape

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        """call the op

        Parameters
        ----------
        q: Tensor, shape(B, M0, H, K0)
        k: Tensor, shape(B, N0, H, K0)
        v: Tensor, shape(B, N0, H, N1)
        bias: Tensor, shape(B, H, M0, N0), optional

        Returns
        ----------
        Tensor, shape(B, H, M0, N1)
        """

        if bias is not None:
            self._attrs["inputs"] = [q, k, v, bias]
        else:
            self._attrs["inputs"] = [q, k, v]
        self._set_depth()
        output_shape = self._infer_shapes()
        self._check_alignment()
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        o_shape = [var.upper_bound() for var in output_shape]
        if o_shape[-1] > 128:
            self._attrs["workspace"] = 4 * np.prod(o_shape)

        return output

    def _get_op_attributes(self):
        target_attrs = [
            "causal_type",
            "epilogue_math_name",
            "alpha0",
            "alpha1",
            "alpha1_divide_by_seq_len",
        ]
        attr = {}

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr

    def gen_function(self) -> str:
        """call backend functions"""
        current_target = target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=current_target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
