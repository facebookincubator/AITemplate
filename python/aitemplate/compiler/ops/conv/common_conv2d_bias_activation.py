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
Fused conv2d_bias_activation op.
"""
from typing import Tuple

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.conv.conv2d import conv2d


# pylint: disable=C0103
class conv2d_bias_activation(conv2d):
    """Base class of conv2d with bias + activation."""

    def __init__(self, activation, stride, pad, dilate=1, group=1) -> None:
        """Conv2d_bias_activation constructor.

        Parameters
        ----------
        activation : str
            The name of the activation operator
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
        self._attrs["op"] = "conv2d_bias_{act}".format(act=activation)
        self._attrs["epilogue"] = "LinearCombinationRelu"

    def __call__(self, x: Tensor, w: Tensor, b: Tensor):
        """Call conv2d_bias_activation with tensors x, w, b

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

    def _get_op_attributes(self):
        attrs = super()._get_op_attributes()
        attrs.update({"activation": self._attrs["op"].split("_")[-1]})

        return attrs

    @staticmethod
    def is_valid_inputs(x: Tensor, w: Tensor, b: Tensor) -> Tuple[bool, str]:
        x_shape = x._attrs["shape"]
        if len(x_shape) != 4:
            return False, f"x should be 4D: {x_shape=}"

        w_shape = w._attrs["shape"]
        if len(w_shape) != 4:
            return False, f"w should be 4D: {w_shape=}"

        b_shape = b._attrs["shape"]
        if len(b_shape) != 1:
            return False, f"b should be 1D: {b_shape=}"

        if b_shape[0] != w_shape[0]:
            return (
                False,
                f"out channels in bias does not match: {b_shape[0]=} != {w_shape[0]=}",
            )

        # No need to check compatibility of x/w. This function is only used
        # for fusing conv/elementwise into conv_bias. If x and w were not compatible,
        # it would fail in the original conv.__call__.
        return True, ""
