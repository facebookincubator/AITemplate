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
patch_embed Module.
"""
from typing import Callable, Tuple

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.conv3d import Conv3d
from aitemplate.frontend.nn.module import Module


class PatchEmbed(Module):
    """
    Transformer basic patch embedding module. Performs patchifying input and flatten

    ::

                                       PatchModel
                                           ↓
                                        flatten

    ::

    output shape: [N, D*H*W, C]

    """

    def __init__(
        self,
        patch_model,
    ) -> None:
        super().__init__()
        self.patch_model = patch_model

    def forward(self, *args) -> Tensor:
        assert len(args) == 1
        x = args[0]

        x = self.patch_model(x)
        x = ops.flatten(start_dim=1, end_dim=-2)(x)
        return x


def create_conv_patch_embed(
    *,
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (1, 16, 16),
    conv_stride: Tuple[int] = (1, 4, 4),
    conv_padding: Tuple[int] = (1, 7, 7),
    conv_bias: bool = True,
    conv: Callable = Conv3d,
) -> Module:
    """
    Creates the transformer basic patch embedding. It performs Convolution, flatten and
    transpose.

    ::

                                        Conv3d
                                           ↓
                                        flatten
                                           ↓
                                       transpose

    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).
        conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
            output.
        conv (callable): Callable used to build the convolution layer.

    Returns:
        (nn.Module): transformer patch embedding layer.
    """
    conv_module = conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        bias=conv_bias,
    )
    return PatchEmbed(patch_model=conv_module)
