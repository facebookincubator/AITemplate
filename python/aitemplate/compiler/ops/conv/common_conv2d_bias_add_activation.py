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
Fused conv2d_bias_add_activation op.
"""

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.conv.conv2d import conv2d


# pylint: disable=C0103
class conv2d_bias_add_activation(conv2d):
    """Base class of conv2d with bias + add + activation."""

    def __init__(self, activation, stride, pad, dilate=1, group=1) -> None:
        """Conv2d_bias_add_activation constructor.

        Parameters
        ----------
        activation : str
            Name of the activation operator
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
        self._attrs["op"] = "conv2d_bias_add_{act}".format(act=activation)
        self._attrs["epilogue"] = "LinearCombinationResidualBlock"

    def __call__(self, x: Tensor, w: Tensor, b: Tensor, r: Tensor):
        """Call conv2d_bias_add_activation with tensors x, w, b, r

        Parameters
        ----------
        x : Tensor
            in shape (N, H, W, C_in)
        w : Tensor
            in shape (C_out, K_h, K_w, C_in)
        b : Tensor
            in shape (C_out)
        r : Tensor
            in shape (N, H_out, W_out, C_out)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, H_out, W_out, C_out)
        """
        self._attrs["inputs"] = [x, w, b, r]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        attrs = super()._get_op_attributes()
        attrs.update({"activation": self._attrs["op"].split("_")[-1]})

        return attrs
