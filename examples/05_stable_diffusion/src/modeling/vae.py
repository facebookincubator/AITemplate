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
"""
Translated from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/vae.py.
"""

from typing import Tuple

from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor

from .unet_blocks import get_down_block, get_up_block, UNetMidBlock2D
from aitemplate.testing import detect_target

class Decoder(nn.Module):
    def __init__(
        self,
        batch_size,
        height,
        width,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        act_fn="silu",
        dtype="float16",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2dBias(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
        )

        # mid
        self.mid_block = UNetMidBlock2D(
            batch_size,
            height,
            width,
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=32,
            temb_channels=None,
            dtype=dtype,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                attn_num_head_channels=None,
                dtype=dtype,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = 32
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=num_groups_out,
            eps=1e-6,
            use_swish=True,
            dtype=dtype,
        )
        self.conv_out = nn.Conv2dBias(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            dtype=dtype,
        )

    def forward(self, z) -> Tensor:
        sample = z
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_out(sample)

        return sample


class Encoder(nn.Module):
    def __init__(
        self,
        batch_size,
        height,
        width,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
        dtype="float16",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        conv_op = (
            nn.Conv2dBiasFewChannels
            if detect_target().name() == "cuda"
            else nn.Conv2dBias
        )
        self.conv_in = conv_op(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
                dtype=dtype,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            batch_size,
            height,
            width,
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
            dtype=dtype,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
            dtype=dtype,
        )
        self.conv_act = ops.silu

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2dBias(
            block_out_channels[-1],
            conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
        )

    def forward(self, x):
        sample = x

        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        batch_size: int,
        height: int,
        width: int,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        dtype="float16",
    ):
        super().__init__()
        self.decoder = Decoder(
            batch_size,
            height,
            width,
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            dtype=dtype,
        )
        self.post_quant_conv = nn.Conv2dBias(
            latent_channels,
            latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
        )

        self.encoder = Encoder(
            batch_size,
            height,
            width,
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            dtype=dtype,
        )
        self.quant_conv = nn.Conv2dBias(
            2 * latent_channels,
            2 * latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
        )

    def decode(self, z: Tensor, return_dict: bool = True):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec._attrs["is_output"] = True
        dec._attrs["name"] = "pixels"
        return dec

    def encode(
        self,
        x: Tensor,
        sample: Tensor = None,
        return_dict: bool = True,
        deterministic: bool = False,
    ):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if sample is None:
            return moments
        mean, logvar = ops.chunk()(moments, 2, dim=3)
        logvar = ops.clamp()(logvar, -30.0, 20.0)
        std = ops.exp(0.5 * logvar)
        # var = ops.exp(logvar)
        # if deterministic:
        #     var = std = Tensor(mean.shape(), value=0.0, dtype=mean._attrs["dtype"])
        sample._attrs["shape"] = mean._attrs["shape"]
        std._attrs["shape"] = mean._attrs["shape"]
        z = mean + std * sample
        z._attrs["is_output"] = True
        z._attrs["name"] = "latent"
        return z
