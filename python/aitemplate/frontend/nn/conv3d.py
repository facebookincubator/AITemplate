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
conv3d Module.
"""
from ...compiler.ops import conv3d, conv3d_bias, depthwise_conv3d
from .module import Module
from .parameter import Parameter

# pylint: disable=C0103


class Conv3d(Module):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of padding applied to the input.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or Tuple(int)): Size of the convolving kernel
        stride (int or Tuple(int)): Stride of the convolution
        padding (int or Tuple(int), optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or Tuple(int), optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        dtype (string, optional): Data type. Default: "float16"
        bias (bool, optional): Has bias or not. Default: False (Note that we only support bias for depthwise_conv3d for now)

    Shape:
        - Input: :math:`(N, D_{in}, H_{in}, W_{in}, C_{in})`
        - Output: :math:`(N, D_{out}, H_{out}, W_{out}, C_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in}  + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out_channels}, \text{kernel_size}[0], \text{kernel_size}[1], \text{kernel_size}[2], `
            :math:`\frac{\text{in_channels}}{\text{groups}})`.

    Examples::

        >>> m = nn.Conv3d(16, 33, 3, 2)
        >>> input = Tensor(shape=[20, 50, 100, 100, 16])
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
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
        bias=False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.weight = Parameter(
            shape=[out_channels, *kernel_size, in_channels // groups], dtype=dtype
        )
        if groups != 1 and bias:
            self.bias = Parameter(shape=[out_channels], dtype=dtype)

        if groups == 1:
            if bias:
                self.op = conv3d_bias(
                    stride=stride, pad=padding, dilate=dilation, group=groups
                )
            self.op = conv3d(stride=stride, pad=padding, dilate=dilation, group=groups)
        else:
            self.op = depthwise_conv3d(
                stride=stride, pad=padding, dilate=dilation, group=groups, bias=bias
            )

    def forward(self, *args):
        """Applies Conv3d on the input tensor."""
        assert len(args) == 1
        x = args[0]
        return self.op(x, self.weight.tensor())
