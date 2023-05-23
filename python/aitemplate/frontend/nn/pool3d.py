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
pool3d-family modules.
"""
from aitemplate.compiler.ops import max_pool2d
from aitemplate.compiler.ops.common import reshape
from aitemplate.frontend.nn.module import Module


def identical_elem_tuple_to_int(param):
    """
    Convert tuples with all the same int elem to
    a single int (ex. (3, 3, 3) --> 3)
    """
    if isinstance(param, int):
        return param

    if not isinstance(param, (list, tuple)) or not all(x == param[0] for x in param):
        raise RuntimeError(f"AIT supports square param values only, but got {param}")
    return param[0]


class MaxPool3d(Module):
    r"""Applies a 3D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, D, H, W, C)`,
    output :math:`(N, D_{out}, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, d, h, w, C_j) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window
        padding: implicit zero padding to be added on both sides
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, *args):
        assert len(args) == 1
        input_val = args[0]

        if (
            isinstance(self.kernel_size, tuple)
            and isinstance(self.stride, tuple)
            and isinstance(self.padding, tuple)
        ):
            kernel_size_tuple = self.kernel_size
            stride_tuple = self.stride
            padding_tuple = self.padding

            assert (
                kernel_size_tuple[0] == 1
            ), "max_pool3d only supports kT == 1 currently"
            assert stride_tuple[0] == 1, "max_pool3d only supports sT == 1 currently"
            assert (
                padding_tuple[0] == 0
            ), "max_pool3d only supports T_padding == 0 currently"

            kernel_size = identical_elem_tuple_to_int(kernel_size_tuple[1:])
            stride = identical_elem_tuple_to_int(stride_tuple[1:])
            padding = identical_elem_tuple_to_int(padding_tuple[1:])
        elif (
            isinstance(self.kernel_size, int)
            and isinstance(self.stride, int)
            and isinstance(self.padding, int)
        ):
            kernel_size = self.kernel_size
            stride = self.stride
            padding = self.padding
        else:
            raise RuntimeError("Only int or tuple types are supported")

        N, D, H, W, C = input_val.shape()

        reshape_op_0 = reshape()
        shape_0 = (-1, H, W, C)
        input_val = reshape_op_0(input_val, shape_0)

        output = max_pool2d(kernel_size=kernel_size, stride=stride, pad=padding)(
            input_val
        )

        _, H_o, W_o, _ = output.shape()
        reshape_op_1 = reshape()
        shape_1 = (N, D, H_o, W_o, C)

        output = reshape_op_1(output, shape_1)
        return output
