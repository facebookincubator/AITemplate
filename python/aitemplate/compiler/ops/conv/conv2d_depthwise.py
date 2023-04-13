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
Fused conv2d_depthwise op.
"""
from typing import List, Tuple

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.conv.conv2d import conv2d


# pylint: disable=C0103
class conv2d_depthwise(conv2d):
    """Base class of conv2d with groups."""

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """conv2d_depthwise constructor.

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
        self._attrs["op"] = "conv2d_depthwise"

    def __call__(self, x: Tensor, w: Tensor):
        """Call conv2d_depthwise with tensors x, w

        Parameters
        ----------
        x : Tensor
            in shape (N, H, W, C_in)
        w : Tensor
            in shape (C_out, K_h, K_w, 1)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, H_out, W_out, C_out)
        """
        self._attrs["inputs"] = [x, w]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        output = Tensor(output_shape, src_ops={self})
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        self._attrs["outputs"] = [output]
        return output

    def _infer_shape(self, x: List[int], w: List[int]) -> List[int]:
        if w[0] != self._attrs["group"]:
            raise RuntimeError("W Shape mismatch for conv2d_depthwise")
        return super()._infer_shape(x, w)

    @staticmethod
    def is_valid_inputs(x: Tensor, w: Tensor) -> Tuple[bool, str]:
        x_shape = x._attrs["shape"]
        if len(x_shape) != 4:
            return False, f"x should be 4D: {x_shape=}"

        w_shape = w._attrs["shape"]
        if len(w_shape) != 4:
            return False, f"w should be 4D: {w_shape=}"

        # No need to check compatibility of x/w. This function is only used
        # for fusing conv/elementwise into conv_bias. If x and w were not compatible,
        # it would fail in the original conv.__call__.
        return True, ""
