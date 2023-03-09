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
Avg_pool2d op.
"""
from aitemplate.compiler.ops.pool.pool2d import pool2d_base


# pylint: disable=C0103
class avg_pool2d(pool2d_base):
    r"""Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, H, W, C)`,
    output :math:`(N, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`pad` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`pad` number of points.

    * .attr.:`kernel_size`: the size of the window

    * .attr.:`stride`: the stride of the window

    * .attr.:`pad`: implicit zero padding to be added on both sides

    Args:
        input (Tensor [N, H, W, C]): the input tensor.

    Return:
        Tensor [N, H_out, W_out, C].
    """

    def __init__(self, kernel_size, stride, pad) -> None:
        super().__init__(stride, pad, kernel_size, "avg")
        self._attrs["op"] = "avg_pool2d"
