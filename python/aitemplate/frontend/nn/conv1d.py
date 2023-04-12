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
Conv1d Module.
"""
from aitemplate.compiler.ops import conv2d, conv2d_bias, squeeze, unsqueeze
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter


class Conv1d(Module):
    r"""
    Conv1d module applies a 1D convolution over an input signal composed of several input planes.

    .. math::
        \text{out}\left(B_i, \text{:}, \text{channels\_out}_j\right) = \text{bias}\left(\text{channels\_out}_j\right) +
        \sum_{k = 0}^{\text{channels\_in} - 1} \text{weight}\left(\text{channels\_out}_j, \text{:}, k\right)
        \star \text{input}\left(B_i, \text{:}, k\right)

    The semantics are similar to `PyTorch`_ with the following exception:
    dims 1 and 2 of the weight, input and output are swapped (while dim 0 remains the same).

    .. _PyTorch:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype: str = "float16",
        bias: bool = False,
        name: str = "conv1d",
    ):
        super().__init__()

        self.weight = Parameter(
            shape=[out_channels, kernel_size, in_channels // groups],
            dtype=dtype,
            name=f"{name}_weight",
        )
        if bias:
            self.bias = Parameter(
                shape=[out_channels], dtype=dtype, name=f"{name}_bias"
            )
        else:
            self.bias = None

        # note that conv1d is functionally equivalent to conv2d,
        # but we need to reshape the input, weight and output tensors,
        # as well as use the correct stride, padding and dilation for the conv2d op.
        fwd_func = conv2d_bias if bias else conv2d
        self.op = fwd_func(
            stride=(stride, 1), pad=(padding, 0), dilate=(dilation, 1), group=groups
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Applies Conv1d on the input tensor of shape :math:`(B, \text{seq\_in}, \text{channels\_in})`.
        The output has shape :math:`(B, \text{seq\_out}, \text{channels\_out})`, where
        .. math::
            \text{seq\_out} = \left\lfloor\frac{\text{seq\_in} + 2 \times \text{padding} - \text{dilation}
                             \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
        """
        # make the conv2d inputs 4d
        xu = unsqueeze(dim=2)(x)
        wu = unsqueeze(dim=2)(self.weight.tensor())
        if self.bias is None:
            c2d = self.op(xu, wu)
        else:
            c2d = self.op(xu, wu, self.bias.tensor())
        # make the result 3d again
        return squeeze(dim=2)(c2d)
