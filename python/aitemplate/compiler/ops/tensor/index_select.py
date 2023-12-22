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
Define masked_select op
"""

from typing import List

from aitemplate.backend import registry

from aitemplate.backend.target import Target

from aitemplate.compiler.base import Operator, Tensor


class index_select(Operator):
    """
    Returns a new tensor which indexes the input tensor
    along dimension dim using the entries in index which is a LongTensor.

    The returned tensor has the same number of dimensions as the original tensor (input).
    The dimth dimension has the same size as the length of index;
    other dimensions have the same size as in the original tensor.

    Args:
        input (Tensor) – the input tensor.
        dim (int) – the dimension in which we index
        index (IntTensor or LongTensor) – the 1-D tensor containing the indices to index
    """

    def __init__(self, dim=0):
        super().__init__()
        self._attrs["op"] = "index_select"
        self._attrs["dim"] = dim

    def _normalize_dim(self, rank: int):
        dim_idx = self._attrs["dim"]
        orig = dim_idx
        if dim_idx < 0:
            dim_idx = rank + dim_idx
        if dim_idx < 0 or dim_idx >= rank:
            raise RuntimeError(
                f"Invalid dim for index_select. Valid values of dim range from {-rank} to {rank - 1}. {orig} provided, normalized {dim_idx}"
            )
        self._attrs["dim"] = dim_idx

    def _infer_shape(self, x: Tensor, idx_select_dim):
        self._normalize_dim(len(x._attrs["shape"]))
        dim_idx = self._attrs["dim"]
        dims = x._attrs["shape"][:dim_idx]
        dims += [idx_select_dim]
        if dim_idx + 1 < len(x._attrs["shape"]):
            dims += x._attrs["shape"][dim_idx + 1 :]
        return dims

    def __call__(
        self,
        x: Tensor,
        dim_idxs: Tensor,
    ) -> List[Tensor]:
        self._attrs["inputs"] = [x, dim_idxs]
        if len(dim_idxs._attrs["shape"]) != 1:
            raise RuntimeError("index tensor must be 1 dimensional.")
        self._set_depth()
        output = Tensor(
            self._infer_shape(x, (dim_idxs._attrs["shape"][0])),
            src_ops={self},
            dtype=x._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
