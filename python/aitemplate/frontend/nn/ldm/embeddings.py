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
import math

from aitemplate.compiler import ops
from aitemplate.frontend import nn


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim, specialization="swish")
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = 1
        self.max_period = 10000
        half_dim = self.num_channels // 2
        self.arange = nn.Parameter(shape=[half_dim], dtype="float16", name="arange")

    def forward(self, timesteps):
        assert len(get_shape(timesteps)) == 1, "Timesteps should be a 1d-array"

        half_dim = self.num_channels // 2

        exponent = (-math.log(self.max_period)) * self.arange.tensor()
        exponent = exponent * (1.0 / (half_dim - self.downscale_freq_shift))

        emb = ops.exp(exponent)
        emb = ops.reshape()(timesteps, [-1, 1]) * ops.reshape()(emb, [1, -1])

        # scale embeddings
        emb = self.scale * emb

        # concat sine and cosine embeddings
        if self.flip_sin_to_cos:
            emb = ops.concatenate()(
                [ops.cos(emb), ops.sin(emb)],
                dim=-1,
            )
        else:
            emb = ops.concatenate()(
                [ops.sin(emb), ops.cos(emb)],
                dim=-1,
            )
        return emb
