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
positional_encoding Modules.
"""
import logging
from typing import Tuple

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter

_LOGGER = logging.getLogger(__name__)

# These op implementations are copied from: https://fburl.com/code/o0qhusw6.
# TODO: Move these to proper AIT op FEs
def tile(input_val, dims):
    shape_dims = list(dims)
    input_dim_len = len(input_val.shape())
    result = input_val
    if len(shape_dims) < input_dim_len:
        for _ in range(input_dim_len - len(shape_dims)):
            shape_dims.insert(0, 1)
    if input_dim_len < len(shape_dims):
        shape = input_val.shape()
        for _ in range(len(shape_dims) - input_dim_len):
            shape.insert(0, IntImm(1))
        result = ops.expand()(input_val, shape)

    for i, shape in enumerate(shape_dims):
        # Avoid operate on batch_size dim
        if input_val.shape()[i]._attrs["name"] is not None:
            continue
        cat_groups = [result] * shape
        result = ops.concatenate()(cat_groups, dim=i)
    return result


def repeat(input_val, dims):
    if (
        isinstance(dims, (list, tuple))
        and len(dims) > 0
        and not all(isinstance(x, int) for x in dims)
    ):
        _LOGGER.info("Not mapping repeat to an op. We can't handle variable dims.")
        return input_val
    return tile(input_val, dims)


def repeat_interleave(input_val, repeats, dim=None):
    if not (type(repeats) is int):
        _LOGGER.info(
            "Not mapping repeat_interleave to an acc op. We currently only support `repeat_interleave` with int repeats"
        )
        return
    assert (
        type(repeats) is int
    ), "We currently only support `repeat_interleave` with int repeats"
    rank = len(input_val.shape())
    if dim is None:
        repeat_dim = rank - 1
    else:
        assert type(dim) is int, "dim should be an int"
        repeat_dim = dim
    tile_dims = [1] * (rank + 1)
    tile_dims[repeat_dim + 1] = repeats

    x = ops.unsqueeze(repeat_dim + 1)(input_val)
    x = tile(x, tuple(tile_dims))
    new_shape = []
    if dim is not None:
        if dim < 0:
            repeat_dim = dim + rank
        else:
            repeat_dim = dim
        size_node = input_val.shape()
        for i in range(rank):
            shape_i = ops.getitem()(size_node, i)
            if i == repeat_dim:
                new_shape.append(-1)
            else:
                new_shape.append(shape_i)
    else:
        new_shape.append(-1)

    x = ops.reshape()(x, new_shape)
    return x


class SpatioTemporalClsPositionalEncoding(Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
        sep_pos_embed: bool = False,
        has_cls: bool = True,
        dtype: str = "float16",
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert (
            len(patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.cls_embed_on = has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = tuple(patch_embed_shape)
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.cls_embed_on:
            self.cls_token = Parameter(shape=[1, 1, embed_dim], dtype=dtype)
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            self.cls_token = Parameter(shape=[], value=0, dtype=dtype)
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_spatial = Parameter(
                shape=[1, self.num_spatial_patch, embed_dim],
                dtype=dtype,
            )
            self.pos_embed_temporal = Parameter(
                shape=[1, self.num_temporal_patch, embed_dim],
                dtype=dtype,
            )
            if self.cls_embed_on:
                self.pos_embed_class = Parameter(shape=[1, 1, embed_dim], dtype=dtype)
            else:
                self.pos_embed_class = Parameter(shape=[], dtype=dtype)
            self.pos_embed = Parameter(shape=[], dtype=dtype)

        else:
            self.pos_embed = Parameter(shape=[1, num_patches, embed_dim], dtype=dtype)
            # Placeholders for torchscriptability, won't be used
            self.pos_embed_spatial = Parameter(shape=[], dtype=dtype)
            self.pos_embed_temporal = Parameter(shape=[], dtype=dtype)
            self.pos_embed_class = Parameter(shape=[], dtype=dtype)

    def patch_embed_shape(self) -> Tuple[int, int, int]:
        return self._patch_embed_shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor.
        """
        B, N, C = x.shape()
        if self.cls_embed_on:
            cls_tokens = ops.expand()(self.cls_token.tensor(), [B, -1, -1])
            x = ops.concatenate()([cls_tokens, x], dim=1)

        if self.sep_pos_embed:
            pos_embed = ops.elementwise(FuncEnum.ADD)(
                repeat(
                    self.pos_embed_spatial.tensor(), (1, self.num_temporal_patch, 1)
                ),
                repeat_interleave(
                    self.pos_embed_temporal.tensor(), self.num_spatial_patch, dim=1
                ),
            )

            if self.cls_embed_on:
                pos_embed = ops.concatenate()(
                    [self.pos_embed_class.tensor(), pos_embed], dim=1
                )
            x = ops.elementwise(FuncEnum.ADD)(x, pos_embed)
        else:
            x = ops.elementwise(FuncEnum.ADD)(x, self.pos_embed.tensor())

        return x
