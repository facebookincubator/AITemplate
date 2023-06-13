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

Notation:
B: batch size
H: number of heads

If inputs/outputs have three dims ( singlehead case ):
Q: [B, M0, K0] (row_major),
K: [B, N0, K0] (column_major),
V: [B, N0, N1] (row_major),
bias: [B, M0, N0] (row_major).
output: [ B, M0, N1 ]

If inputs/outputs have four dims ( multihead case ),
the head dim is located at the dimension with index 2

dimension order of the parameters is

Q: [B, M0, H, K0] (row_major),
K: [B, N0, H, K0] (column_major),
V: [B, N0, H, N1] (row_major),
bias: [B, H, M0, N0] (row_major).
Output: [ B, M0, H, N1 ]

Only supports NO_CAUSAL or LOWER_LEFT_EMPTY causal mask types.
When causal_mask is enabled, M0 must be equal to N0.

Internally, it stores the results of Q@K in registers without writing them to shared memory, which is faster.
However, N0 and N1 must be <= 512.
"""

from aitemplate.backend import registry, target
from aitemplate.compiler.base import IntVar, Tensor
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import b2b_bmm_base, CausalType
from aitemplate.utils import shape_utils


def _is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


class grouped_classic_b2b_bmm(b2b_bmm_base):
    def __init__(
        self,
        causal_type: CausalType,
        epilogue_math_name: str,
        alpha0: float,
        alpha1: float,
        alpha1_divide_by_seq_len: bool = False,
    ) -> None:
        r"""Back-to-back batched gemm fused kernels.

        More detailed documentation at the top of this file.

        Args:
        * causal_type (CausalType): Type of causal_mask. See comments above.
        * epilogue_math_name (str): Name of the activation function.
        Supported epilogue functions can be found from
        python/aitemplate/utils/mk_cutlass_lib/extra_enum.py.
        * alpha0 (float): See the math function above.
        * alpha1 (float): See the math function above.
        * alpha1_divide_by_seq_len (bool) Whether divide alpha1 by seq_len.
        Useful when seq_len is a dynamic value so that alpah1 cannot be
        computed in advance.
        """
        super().__init__(
            causal_type, epilogue_math_name, alpha0, alpha1, alpha1_divide_by_seq_len
        )
        self._attrs["op"] = "grouped_classic_b2b_bmm"
        if (
            causal_type != CausalType.NO_CAUSAL
            and causal_type != CausalType.LOWER_LEFT_EMPTY
        ):
            raise NotImplementedError(
                f"grouped_classic_b2b_bmm only supports NO_CAUSAL or LOWER_LEFT_EMPTY. Current causal type: {causal_type}"
            )

    def _infer_shapes(self):
        """infer the output shape for grouped_classic_b2b_bmm."""
        q, k, v, bias = self._attrs["inputs"]
        if not (q.is_jagged() and k.is_jagged() and v.is_jagged()):
            raise RuntimeError(f"{q=}, {k=}, {v=} must be jagged!")
        q_shape = q._attrs["shape"]
        k_shape = k._attrs["shape"]
        v_shape = v._attrs["shape"]
        bias_shape = bias._attrs["shape"]
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
        if q_shape[1] != k_shape[1] or q_shape[1] != v_shape[1]:
            raise RuntimeError(
                f"QKV must have same head size! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )

        if q_shape[2] != k_shape[2]:
            raise RuntimeError(
                f"Q and K shapes are not compatible ( inner dimension for Matmul must be identical ) - Q shape: {q_shape=}, K shape: {k_shape=}."
            )

        batch_size = q_shape[0]
        K0 = q_shape[-1]
        if K0 != k_shape[-1]:
            raise RuntimeError(
                f"Q and K shapes are not compatible! QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}."
            )

        num_heads = q_shape[1]
        output_shape = [q_shape[0], num_heads, v_shape[2]]

        batch_size = q_shape[0].batch_dim()
        max_seq_len = q_shape[0].jagged_dims()[0].max_value()
        if isinstance(max_seq_len, IntVar):
            if max_seq_len.lower_bound() != max_seq_len.upper_bound():
                raise RuntimeError(
                    "Maximum sequence length needs to be a fixed (IntImm) dimension. "
                )
            max_seq_len = max_seq_len.upper_bound()

        # This is a current limitation of the classic op due to grid layout and test results
        if (
            (not _is_power_of_two(max_seq_len))
            or (max_seq_len > 512)
            or (max_seq_len < 64)
        ):
            raise RuntimeError(
                f"Maximum sequence length needs to be a fixed (IntImm) dimension with a power of two between 64 and 512 for the grouped classic b2b op to work. Actual value: {max_seq_len=}. {type(max_seq_len)=}"
            )
        if len(bias_shape) != 4:
            raise RuntimeError(f"Expected bias rank 4. Current bias rank: {len(bias)}.")

        bias_expected_shape = [
            batch_size,
            num_heads,
            max_seq_len,
            max_seq_len,
        ]
        broadcastable, _ = shape_utils.get_broadcast_max_shape(
            bias_shape, bias_expected_shape
        )
        if not broadcastable:
            raise RuntimeError(
                f"bias shape is not compatible with Q K! "
                f"QKV shapes: {q_shape=}, {k_shape=}, {v_shape=}, "
                f"bias shapes: {bias_shape=}, {bias_expected_shape=}."
            )
        if bias_shape[-1] != bias_expected_shape[-1]:
            raise RuntimeError(
                f"Bias last dim is not broadcastable! Expected shape: {bias_expected_shape[-1]}, current bias shape: {bias_shape}"
            )
        return output_shape, max_seq_len

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """call the op

        Note: [H,] means optional num-heads,
        if it exists for one input tensor, all need to have it,
        Parameters
        ----------
        q: Tensor, shape(B, M0, [H,] K0)
        k: Tensor, shape(B, N0, [H,] K0)
        v: Tensor, shape(B, N0, [H,] N1)
        bias: Tensor, shape(B, [H,] M0, N0)

        Returns
        ----------
        Tensor, shape(B, M0, [H,], N1)
        """

        self._attrs["inputs"] = [q, k, v, bias]
        self._set_depth()
        output_shape, max_seq_len = self._infer_shapes()
        self._check_alignment()
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        self._attrs["max_seq_len"] = max_seq_len

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
                "grouped_classic_b2b_bmm is only supported by CUDA>=SM80 devices."
            )
        func_key = "{target}.{op}.gen_function".format(
            target=current_target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
