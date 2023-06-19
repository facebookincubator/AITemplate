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
Define jagged_lengths_to_presences op
"""
from typing import List

from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.dtype import get_dtype_size


class jagged_lengths_to_presences(Operator):
    """
    Given a 1D Tensor of lengths of the sequences in a jagged Tensor,
    returns a 2D Tensor of presences indicating where the data exists
    and where not. The dtype of presences Tensor is configurable.

    Args:
        lengths (Tensor): 1D Tensor of sequence lengths, [B]-shaped.
        max_seq_len (int): Maximum possible sequence length.
    Returns:
        presences (Tensor): 2D Tensor of presences, [B, max_seq_len]-shaped.
                            presences[i, j] = (dtype)(j < lenghts[i])
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "jagged_lengths_to_presences"
        self._attrs["has_profiler"] = False

    def _infer_shape(
        self,
        lengths: Tensor,
        max_seq_len: int,
    ) -> List[IntVar]:
        batch_size = lengths.shape()[0]
        return [batch_size, IntImm(max_seq_len)]

    def __call__(
        self,
        lengths: Tensor,
        max_seq_len: int,
        dtype: str = "bool",
    ) -> Tensor:
        if len(lengths.shape()) != 1:
            raise ValueError(f"The lengths Tensor must be 1D, but got {lengths=}.")
        if lengths._attrs["dtype"] not in ("int32", "int64"):
            raise ValueError(
                f"The lengths Tensor must be int32 or int64, but got {lengths=}."
            )
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive integer, but got {max_seq_len=}."
            )

        # validation inside
        get_dtype_size(dtype)

        self._attrs["inputs"] = [lengths]
        self._set_depth()

        output_shape = self._infer_shape(lengths, max_seq_len)
        presences = Tensor(
            output_shape,
            src_ops={self},
            dtype=dtype,
        )

        self._attrs["outputs"] = [presences]
        return presences

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
