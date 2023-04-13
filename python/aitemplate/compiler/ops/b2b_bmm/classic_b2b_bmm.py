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
Back-to-back batched gemm fused kernel.
Computes bmm(causal_mask(alpha1 * (activation(alpha0 * bmm(Q, K) + bias))), V),

where:
Q: [B, M0, K0] (row_major),
K: [B, N0, K0] (column_major),
V: [B, N0, N1] (row_major),
bias: [B, M0, N0] (row_major).
Layouts are fixed for now.

Only supports NO_CAUSAL or LOWER_LEFT_EMPTY for now.
When causal_mask is enabled, M0 must be equal to N0.

Internally, it stores the results of Q@K in registers without writing them to shared memory, which is faster.
However, N0 / N1 must be <= 512.
"""

from aitemplate.backend import registry, target
from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import b2b_bmm_base, CausalType


class classic_b2b_bmm(b2b_bmm_base):
    """See comments at the head of this file."""

    def __init__(
        self,
        causal_type: CausalType,
        epilogue_math_name: str,
        alpha0: float,
        alpha1: float,
        alpha1_divide_by_seq_len: bool = False,
    ) -> None:
        """Initialize classic_b2b_bmm op.
        Check aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base for more details
        about these args.
        """
        super().__init__(
            causal_type, epilogue_math_name, alpha0, alpha1, alpha1_divide_by_seq_len
        )
        self._attrs["op"] = "classic_b2b_bmm"
        if (
            causal_type != CausalType.NO_CAUSAL
            and causal_type != CausalType.LOWER_LEFT_EMPTY
        ):
            raise NotImplementedError(
                f"classic_b2b_bmm only supports NO_CAUSAL or LOWER_LEFT_EMPTY. Current causal type: {causal_type}"
            )

    def _infer_shapes(self):
        """infer the output shape for classic_b2b_bmm."""
        q, k, v, bias = self._attrs["inputs"]
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
                f"QKV must have same batch size! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        batch_size = q_shape[0]
        M0 = q_shape[1]
        K0 = q_shape[2]
        if K0 != k_shape[2]:
            raise RuntimeError(
                f"Q K shapes are not compatible! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        N0 = k_shape[1]
        if N0 != v_shape[1]:
            raise RuntimeError(
                f"K V shapes are not compatible! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )
        N1 = v_shape[2]
        if N0.upper_bound() > 512 or N1.upper_bound() > 512:
            raise RuntimeError(
                f"classic_b2b_bmm only supports <=512 N0 / N1. Current length: {N0=}, {N1=}"
            )
        if not isinstance(N0, IntImm) or not isinstance(N1, IntImm):
            raise RuntimeError(
                f"classic_b2b_bmm only supports static N0 / N1. Current {N0=}, {N1=}."
            )
        if self._attrs["causal_type"] != CausalType.NO_CAUSAL:
            if M0 != N0:
                raise RuntimeError(
                    f"When causal_type is enabled, M0 must be equal to N0. Current {M0=}, {N0=}."
                )

        output_shape = [batch_size, M0, N1]

        bias_shape = bias._attrs["shape"]
        if bias_shape != [batch_size, M0, N0]:
            raise RuntimeError(
                f"bias shape is not compatible with Q K! "
                f"QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}, "
                f"bias shapes: {bias_shape=}."
            )
        return output_shape

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """call the op

        Parameters
        ----------
        q: Tensor, shape(B, M0, K0)
        k: Tensor, shape(B, N0, K0)
        v: Tensor, shape(B, N0, N1)
        bias: Tensor, shape(B, M0, N0)

        Returns
        ----------
        Tensor, shape(B, M0, N1)
        """

        self._attrs["inputs"] = [q, k, v, bias]
        self._set_depth()
        output_shape = self._infer_shapes()
        self._check_alignment()
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]

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
        if current_target.name() == "rocm" or (
            current_target.name() == "cuda" and int(current_target._arch) < 80
        ):
            raise NotImplementedError(
                "classic_b2b_bmm is only supported by CUDA>=SM80 devices."
            )
        func_key = "{target}.{op}.gen_function".format(
            target=current_target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
