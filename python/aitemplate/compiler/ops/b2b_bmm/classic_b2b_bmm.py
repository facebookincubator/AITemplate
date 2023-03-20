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
Computes bmm(causal_masks(alpha1(activation(alpha0 * bmm(Q, K) + beta0 * bias))), V),

where:
Q: [B, M0, K0] (row_major), K: [B, N0, K0] (column_major), V: [B, N0, N1] (row_major), bias: [B, M0, N0] (row_major).
Layouts are fixed for now.

causal_masks can be disabled.
When casual_masks is enabled, only the left bottom triangular part of the matrix is valid,
and the other part is set to 0.

Only supports M0 <= 512.
"""

from aitemplate.backend import registry, target
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.utils.alignment import find_max_alignment, get_alignments


def _check_max_alignment(shape: IntVar, dtype: str, error_msg: str) -> None:
    if not isinstance(shape, IntImm):
        raise RuntimeError(f"{shape=} must be IntImm! ", error_msg)
    res = find_max_alignment(shape.value(), dtype) == max(get_alignments(dtype))
    if not res:
        raise RuntimeError(
            f"{shape=} does not satisfy {dtype=} max alignment requirements! ",
            error_msg,
        )


class classic_b2b_bmm(Operator):
    def __init__(
        self, causal: bool, epilogue_math_name: str, alpha0: float, alpha1: float
    ) -> None:
        """Initialize classic_b2b_bmm op."""
        super().__init__()
        self._attrs["op"] = "classic_b2b_bmm"
        self._attrs["has_profiler"] = False
        self._attrs["causal"] = causal
        self._attrs["alpha0"] = alpha0
        self._attrs["alpha1"] = alpha1

        import cutlass_lib

        if epilogue_math_name not in cutlass_lib.library.EpilogueMathName:
            raise RuntimeError(
                "Unsupported epilogue function! Please check "
                "python/aitemplate/utils/mk_cutlass_lib/extra_enum.py for a list of supported epilogue functions."
            )
        self._attrs["epilogue_math_name"] = epilogue_math_name

    def _check_alignment(self) -> None:
        q, k, v, bias = self._attrs["inputs"]
        if (
            q._attrs["dtype"] != k._attrs["dtype"]
            or q._attrs["dtype"] != v._attrs["dtype"]
        ):
            raise RuntimeError(
                "QKV dtypes must be the same! "
                f"QKV dtypes: {q._attrs['dtype']=}, {k._attrs['dtype']=}, {v._attrs['dtype']=}"
            )
        dtype = q._attrs["dtype"]

        _check_max_alignment(q._attrs["shape"][2], dtype, f"{q._attrs['shape']=}")
        _check_max_alignment(k._attrs["shape"][2], dtype, f"{k._attrs['shape']=}")
        _check_max_alignment(v._attrs["shape"][2], dtype, f"{v._attrs['shape']=}")

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
        if M0.upper_bound() > 512:
            raise RuntimeError(
                f"classic_b2b_bmm only supports <=512 seq_length. Current length: {M0}"
            )
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
        self._check_alignment()
        output_shape = self._infer_shapes()
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]

        return output

    def _get_op_attributes(self):
        target_attrs = ["causal", "epilogue_math_name", "alpha0", "alpha1"]
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
