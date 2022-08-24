# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
summary
"""
import math

from typing import List

from ...base import Tensor
from .split import split


class chunk(split):
    """_summary_

    Parameters
    ----------
    chunk : _type_
        _description_
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "split"

    def __call__(self, input: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
        """
        Attempts to split a tensor into the specified number of chunks

        Parameters
        ----------
        input : Tensor
            the tensor to split
        chunks : int
            number of chunks to return. Must be >= 1
        dim : int, optional
            axes along which to split the tensor, by default 0
        Returns
        -------
        List[Tensor]
            If the tensor size along the given dimesion dim is divisible by chunks,
            all returned chunks will be the same size.
            If the tensor size along the given dimension dim is not divisible by chunks,
            all returned chunks will be the same size, except the last one.
            If such division is not possible,
            this function may return less than the specified number of chunks.
        """
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
