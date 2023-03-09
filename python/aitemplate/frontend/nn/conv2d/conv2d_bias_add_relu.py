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
General template module for conv2e + bias + residual + relu
"""
from aitemplate.frontend.nn.conv2d.common_conv2d_bias_add_act import Conv2dBiasAddAct


class Conv2dBiasAddRelu(Conv2dBiasAddAct):
    r"""Applies 2D convolution with bias + add + relu.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out_channels}, \text{kernel_size}, \text{kernel_size}, `
            :math:`\frac{\text{in_channels}}{\text{groups}})`.
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels).

    Args:
        input (Tensor): the input tensor to apply 2D convolution on.
        residual (Tensor): the residule tensor to add after Conv2dBias.

    Examples::

        >>> m = nn.Conv2dBiasAddRelu(128, 256, 3, 1)
        >>> input = Tensor(shape=[4, 28, 28, 128])
        >>> residual = Tensor(shape=[4, 28, 28, 256])
        >>> output = m(input, residual)
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
            "conv2d_bias_add_relu",
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dtype,
        )
