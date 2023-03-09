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
Max_pool2d op.
"""
from aitemplate.compiler.ops.pool.pool2d import pool2d_base


# pylint: disable=C0103
class max_pool2d(pool2d_base):
    r"""
    Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`pad` is non-zero, then the input is implicitly padded with negative infinity on both sides.

    * .attr.:`kernel_size`: the size of the window

    * .attr.:`stride`: the stride of the window

    * .attr.:`pad`: implicit zero padding to be added on both sides

    Args:
        input (Tensor [N, H, W, C]): the input tensor.

    Return:
        Tensor [N, H_out, W_out, C].

    """

    def __init__(self, kernel_size, stride, pad) -> None:
        super().__init__(stride, pad, kernel_size, "max")
        self._attrs["op"] = "max_pool2d"
