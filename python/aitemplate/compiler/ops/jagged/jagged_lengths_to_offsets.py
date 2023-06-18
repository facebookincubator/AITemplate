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
Define jagged_lengths_to_offsets op
"""
from typing import List

from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.compiler.base import IntVar, Operator, Tensor


class jagged_lengths_to_offsets(Operator):
    """
    Given a 1D Tensor of lengths of the sequences in a jagged Tensor,
    returns the corresponding 1D Tensor of offsets. The latter is the
    inclusive sum of the lengths prepended by a zero.

    Args:
        lengths (Tensor): 1D Tensor of sequence lengths, [B]-shaped.
    Returns:
        offsets (Tensor): 1D Tensor of sequence offsets, [B+1]-shaped.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "jagged_lengths_to_offsets"
        self._attrs["has_profiler"] = False

    def _infer_shape(self, lengths: Tensor) -> List[IntVar]:
        batch_size = lengths.shape()[0]
        # the offsets are 1 element longer than the lengths
        offsets_size = IntVar(
            values=[
                batch_size.lower_bound() + 1,
                batch_size.upper_bound() + 1,
            ]
        )
        return [offsets_size]

    def __call__(
        self,
        lengths: Tensor,
    ) -> Tensor:
        if len(lengths.shape()) != 1:
            raise ValueError(f"The lengths Tensor must be 1D, but got {lengths=}.")
        if lengths._attrs["dtype"] not in ("int32", "int64"):
            raise ValueError(
                f"The lengths Tensor must be int32 or int64, but got {lengths=}."
            )

        self._attrs["inputs"] = [lengths]
        self._set_depth()
        output_shape = self._infer_shape(lengths)
        offsets = Tensor(
            output_shape,
            src_ops={self},
            dtype=lengths._attrs["dtype"],
        )

        # set the workspace to empirically determined large enough value
        sizeof_dtype = 4 if lengths._attrs["dtype"] == "int32" else 8
        self._attrs["workspace"] = max(
            2**16,
            16 * sizeof_dtype * offsets.shape()[0].upper_bound(),
        )

        self._attrs["outputs"] = [offsets]
        return offsets

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
