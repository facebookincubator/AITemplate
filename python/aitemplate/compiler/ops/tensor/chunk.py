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
chunk
"""
import math

from typing import List

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.tensor.split import split


class chunk(split):
    """
    Attempts to split a tensor into the specified number of chunks

    Args:
        input (Tensor): the tensor to split
        chunks (int): number of chunks to return. Must be >= 1
        dim (int) : optional, axes along which to split the tensor, by default 0

    Returns :
        List[Tensor]: If the tensor size along the given dimesion dim is divisible by chunks,
        all returned chunks will be the same size.
        If the tensor size along the given dimension dim is not divisible by chunks,
        all returned chunks will be the same size, except the last one.
        If such division is not possible,
        this function may return less than the specified number of chunks.

    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "split"

    def __call__(self, input: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
        if chunks < 1:
            raise RuntimeError(f"chunks must be >= 1 but got {chunks=}")
        input_shape = input._attrs["shape"]
        input_rank = len(input_shape)
        if input_rank <= 0:
            raise RuntimeError("expected a non-scalar tensor")
        if dim >= input_rank:
            raise RuntimeError(f"chunk {dim=} expected to be less than {input_rank=}")
        split_dim_sizes = input_shape[dim]._attrs["values"]
        if len(split_dim_sizes) > 1:
            raise RuntimeError(f"Not implemented: chunk along dynamic axes {dim=}")
        length = split_dim_sizes[0]
        chunk_size = math.ceil(length / chunks)
        full_chunks = math.floor(length / chunk_size)
        tail_chunk_size = length % chunk_size
        split_size_or_sections = [chunk_size] * full_chunks
        if tail_chunk_size > 0:
            split_size_or_sections.append(tail_chunk_size)
        return super().__call__(
            x=input, split_size_or_sections=split_size_or_sections, dim=dim
        )
