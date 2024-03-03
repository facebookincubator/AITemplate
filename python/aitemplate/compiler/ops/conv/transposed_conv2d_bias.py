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
Fused transposed_conv2d_bias op.
"""

from typing import Tuple

from aitemplate.compiler.base import Tensor

from aitemplate.compiler.ops.conv.common_conv2d_bias_activation import (
    conv2d_bias_activation,
)
from aitemplate.compiler.ops.conv.transposed_conv2d import transposed_conv2d


# pylint: disable=C0103
class transposed_conv2d_bias(transposed_conv2d):
    r"""Transposed conv2d with bias.

    Applies a 2D transposed convolution on input in shape (N, H, W, C_in), adds a bias in shape (C_out) and produces output in shape (N, H_out, W_out, C_out). N is batch size, H, W are the height and width of the input images in pixels, and C is the number of channels.

    Args:
        input: input tensor of shape :math:`(N , H , W, \text{in\_channels})`

        weight: filters of shape :math:`(\text{out\_channels} , K_h, K_w, \frac{\text{in\_channels}}{\text{groups}})`

        bias: optional bias tensor of shape :math:`(\text{out\_channels})`

     This operator uses "channels_last" data format. Below is an example and its equivalence in PyTorch:

    .. highlight:: python
    .. code-block:: python

        X = Tensor(shape=[N, H, W, C_in], dtype="float16", name="images", is_input=True)
        W = Tensor(shape=[C_out, K_h, K_w, C_in], dtype="float16", name="weight", is_input=True)
        B = Tensor(shape=[C_out], dtype="float16", name="bias", is_input=True)
        OP = aitemplate.compiler.ops.transposed_conv2d_bias(stride=1, pad=1, dilate=1)
        Y = OP(X, W, B)


    .. highlight:: python
    .. code-block:: python

        X_pt = NHWC2NCHW(X_ait)
        W_pt = MHWC2NCHW(W_ait)
        B_pt = NHWC2NCHW(B_ait)
        Y_pt = torch.nn.functional.conv_transpose2d(X_pt, W_pt, bias=B_Pt)
        Y = nchw2nhwc(Y_pt)
    """

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """Transposed_conv2d_bias constructor.

        Parameters
        ----------
        stride : int
            Stride of the convolution
        pad : int
            Size of padding to add to the input
        dilate : int, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
           Number of blocked connections from input
            channels to output channels, by default 1
        """
        super().__init__(stride, pad, dilate=dilate, group=group)
        self._attrs["op"] = "transposed_conv2d_bias"
        self._attrs["epilogue"] = "LinearCombination"

    def __call__(self, x: Tensor, w: Tensor, b: Tensor):
        """Call transposed_conv2d_bias with tensors x, w, b.

        Parameters
        ----------
        x : Tensor
            in shape (N, H, W, C_in)
        w : Tensor
            in shape (C_out, K_h, K_w, C_in)
        b : Tensor
            in shape (C_out)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, H_out, W_out, C_out)
        """
        self._attrs["inputs"] = [x, w, b]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        self._attrs["outputs"] = [output]
        return output

    @staticmethod
    def is_valid_inputs(x: Tensor, w: Tensor, b: Tensor) -> Tuple[bool, str]:
        return conv2d_bias_activation.is_valid_inputs(x, w, b)
