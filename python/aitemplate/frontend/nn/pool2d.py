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
pool2d-family modules.
"""
from aitemplate.compiler.ops import avg_pool2d, max_pool2d
from aitemplate.frontend.nn.module import Module


class MaxPool2d(Module):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, H, W, C)`,
    output :math:`(N, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, h, w, C_j) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n, C_j)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window
        padding: implicit zero padding to be added on both sides
    """

    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.op = max_pool2d(kernel_size, stride, padding)

    def forward(self, *args):
        r"""Applies MaxPool2d on the input."""
        assert len(args) == 1
        x = args[0]
        return self.op(x)


class AvgPool2d(Module):
    r"""Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, H, W, C)`,
    output :math:`(N, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, h, w, C_j)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, stride[0] \times h + m, stride[1] \times w + n, C_j)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    Args:
        kernel_size: the size of the window to take an avg over
        stride: the stride of the window
        padding: implicit zero padding to be added on both sides
    """

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.op = avg_pool2d(kernel_size, stride, padding)

    def forward(self, *args):
        r"""Applies AvgPool2d on the input."""
        assert len(args) == 1
        x = args[0]
        return self.op(x)
