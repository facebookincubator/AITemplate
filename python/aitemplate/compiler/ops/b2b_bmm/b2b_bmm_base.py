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
Base class for back-to-back batched gemm fused kernels.
Computes bmm(causal_mask(alpha1 * (activation(alpha0 * bmm(Q, K) + bias))), V),

where:
Q: [B, M0, (H,) K0] (row_major),
K: [B, N0, (H,) K0] (column_major),
V: [B, N0, (H,) N1] (row_major),
bias: [B, (H,) M0, N0] (row_major).
Layouts are fixed for now.
"""

from enum import Enum

from aitemplate.compiler.base import IntImm, IntVar, Operator
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


class CausalType(Enum):
    NO_CAUSAL = 0  # no causal mask
    UPPER_RIGHT_EMPTY = 1  # upper right triangular part of the matrix is 0
    LOWER_LEFT_EMPTY = 2  # bottom left triangular part of the matrix is 0


class b2b_bmm_base(Operator):
    r"""Base class for back-to-back batched gemm fused kernels.

    Computes bmm(causal_mask(alpha1 * (activation(alpha0 * bmm(Q, K) + bias))), V),

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

    def __init__(
        self,
        causal_type: CausalType,
        epilogue_math_name: str,
        alpha0: float,
        alpha1: float,
        alpha1_divide_by_seq_len: bool = False,
    ) -> None:
        """Initialize classic_b2b_bmm op."""
        super().__init__()
        self._attrs["has_profiler"] = False
        self._attrs["causal_type"] = causal_type
        self._attrs["alpha0"] = alpha0
        self._attrs["alpha1"] = alpha1
        self._attrs["alpha1_divide_by_seq_len"] = alpha1_divide_by_seq_len

        import cutlass_lib

        if epilogue_math_name not in cutlass_lib.library.EpilogueMathName:
            raise RuntimeError(
                "Unsupported epilogue function! Please check "
                "python/aitemplate/utils/mk_cutlass_lib/extra_enum.py for a list of supported epilogue functions."
            )
        self._attrs["epilogue_math_name"] = epilogue_math_name

    def _check_alignment(self) -> None:
        q, k, v = self._attrs["inputs"][0:3]
        if (
            q._attrs["dtype"] != k._attrs["dtype"]
            or q._attrs["dtype"] != v._attrs["dtype"]
        ):
            raise RuntimeError(
                "QKV dtypes must be the same! "
                f"QKV dtypes: {q._attrs['dtype']=}, {k._attrs['dtype']=}, {v._attrs['dtype']=}"
            )
        dtype = q._attrs["dtype"]

        _check_max_alignment(q._attrs["shape"][-1], dtype, f"{q._attrs['shape']=}")
        _check_max_alignment(k._attrs["shape"][-1], dtype, f"{k._attrs['shape']=}")
        _check_max_alignment(v._attrs["shape"][-1], dtype, f"{v._attrs['shape']=}")
