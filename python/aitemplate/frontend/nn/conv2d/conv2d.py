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
conv2d Module.
"""
from aitemplate.compiler.ops import conv2d
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter

# pylint: disable=C0103


class Conv2d(Module):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, H, W, C_{\text{in}})` and output :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`H` is a height of input planes in pixels, :math:`W` is
    width in pixels, and :math:`C` denotes a number of channels.


    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of padding applied to the input.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

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

    Examples::

        >>> m = nn.Conv2d(16, 33, 3, 2)
        >>> input = Tensor(shape=[20, 50, 100, 16])
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
    ):
        super().__init__()
        self.weight = Parameter(
            shape=[out_channels, kernel_size, kernel_size, in_channels // groups],
            dtype=dtype,
        )
        self.op = conv2d(stride=stride, pad=padding, dilate=dilation, group=groups)

    def forward(self, *args):
        """Applies Conv2d on the input tensor."""
        assert len(args) == 1
        x = args[0]
        return self.op(x, self.weight.tensor())
