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
conv2d bias module
"""
from aitemplate.frontend.nn.conv2d.common_conv2d_bias_act import Conv2dBiasAct


class Conv2dBias(Conv2dBiasAct):
    r"""Applies 2D convolution with bias.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        dtype (string, optional): Data type. Default: "float16"

    Shape:
        - Input: :math:`(N, H_{in}, W_{in}, C_{in})`
        - Output: :math:`(N, H_{out}, W_{out}, C_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out_channels}, \text{kernel_size}, \text{kernel_size}, `
            :math:`\frac{\text{in_channels}}{\text{groups}})`.
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels).

    Examples::

        >>> m = nn.Conv2d(16, 33, 3, 2)
        >>> input = Tensor(shape=[20, 50, 100, 16])
        >>> output = m(input)

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        dtype="float16",
    ):
        super().__init__(
            "conv2d_bias",
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dtype,
        )
