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
Fused conv2d_bias_relu op.
"""
from aitemplate.compiler.ops.conv.common_conv2d_bias_activation import (
    conv2d_bias_activation,
)


# pylint: disable=C0103
class conv2d_bias_relu(conv2d_bias_activation):
    r"""Conv2d with bias + relu.

    Applies a 2D convolution on input in shape (N, H, W, C_in), adds a bias in shape (C_out), performs relu and produces output in shape (N, H_out, W_out, C_out). N is batch size, H, W are the height and width of the input images in pixels, and C is the number of channels.

    Args:
        input: input tensor of shape :math:`(N , H , W, \text{in\_channels})`

        weight: filters of shape :math:`(\text{out\_channels} , K_h, K_w, \frac{\text{in\_channels}}{\text{groups}})`

        bias: optional bias tensor of shape :math:`(\text{out\_channels})`

    This operator uses "channels_last" data format. Below is an example and its equivalence in PyTorch:

    .. highlight:: python
    .. code-block:: python

        X = Tensor(shape=[N, H, W, C_in], dtype="float16", name="images", is_input=True)
        W = Tensor(shape=[C_out, K_h, K_w, C_in], dtype="float16", name="weight", is_input=True)
        B = Tensor(shape=[C_out], dtype="float16", name="weight", is_input=True)
        OP = aitemplate.compiler.ops.conv2d_bias_relu(stride=1, pad=1, dilate=1)
        Result_ait = OP(X, W, B)

    .. highlight:: python
    .. code-block:: python

        X_pt = NHWC2NCHW(X_ait)
        W_pt = NHWC2NCHW(W_ait)
        B_pt = NHWC2NCHW(B_ait)

        Y = torch.nn.functional.conv2d(X_pt, W_pt, bias=B_pt)
        Result_pt = torch.nn.functional.relu(Y)
        Result_ait = NCHW2NHWC(Result_pt)
    """

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """Conv2d_bias_relu constructor.

        Parameters
        ----------
        stride : int
            Stride of the convolution
        pad : int
            Size of padding to add to the input
        dilate : int, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
            Number of input channels to process to compute one output channel, by default 1
        """
        super().__init__("relu", stride, pad, dilate=dilate, group=group)

    def _get_op_attributes(self):
        attr = super()._get_op_attributes()
        del attr["activation"]

        return attr
