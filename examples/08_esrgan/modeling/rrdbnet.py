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
import torch

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor


def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.op = ops.leaky_relu

    def forward(self, tensor: Tensor) -> Tensor:
        out = self.op(tensor, self.negative_slope)
        return out


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2dBias(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2dBias(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2dBias(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2dBias(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2dBias(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = LeakyReLU(negative_slope=0.2)

    def cat(self, tensors, dim):
        out = ops.concatenate()(tensors, dim)
        return out

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.cat((x, x1), 3)))
        x3 = self.lrelu(self.conv3(self.cat((x, x1, x2), 3)))
        x4 = self.lrelu(self.conv4(self.cat((x, x1, x2, x3), 3)))
        x5 = self.conv5(self.cat((x, x1, x2, x3, x4), 3))
        # Empirically, we use 0.2 to scale the residual for better performance
        out = x5 * 0.2 + x
        return out


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        out = out * 0.2 + x
        return out


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def size(tensor):
    """
    returns maximum of each dynamic dim
    """
    dims = ops.size()(tensor)
    dims = [dim._attrs["int_var"]._attrs["values"][-1] for dim in dims]
    return dims


def pixel_unshuffle(x: Tensor, scale):
    """Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, hh, hw, c).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, hh, hw, c = size(x)
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = ops.reshape()(x, [b, h, scale, w, scale, c])
    return ops.reshape()(
        ops.permute()(x_view, [0, 1, 3, 5, 2, 4]), [b, h, w, out_channel]
    )


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        if num_in_ch < 8:
            self.conv_first = nn.Conv2dBiasFewChannels(num_in_ch, num_feat, 3, 1, 1)
        else:
            self.conv_first = nn.Conv2dBias(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2dBias(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2dBias(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2dBias(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2dBias(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2dBias(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = LeakyReLU(negative_slope=0.2)

    def interpolate(self, tensor):
        op = ops.upsampling2d(scale_factor=2, mode="nearest")
        out = op(tensor)
        return out

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(self.interpolate(feat)))
        feat = self.lrelu(self.conv_up2(self.interpolate(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def map_rrdb(pt_mod, scale=4):
    params_ait = {}
    for key, arr in pt_mod.items():
        arr = arr.to(torch.float16)
        key = key.replace(".", "_")
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        params_ait[key] = arr
    if scale == 4:
        params_ait["conv_first_weight"] = torch.functional.F.pad(
            params_ait["conv_first_weight"], (0, 1)
        )
    return params_ait
