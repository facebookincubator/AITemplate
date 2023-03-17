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
Conv3d with bias.
"""
from typing import List

from aitemplate.compiler.base import Tensor

from aitemplate.compiler.ops.conv.conv3d import conv3d


class conv3d_bias(conv3d):
    r"""conv3d_bias"""

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """Conv3d constructor.

        Parameters
        ----------
        stride : int or tuple
            Stride of the convolution
        pad : int or tuple
            Size of padding to add to the input
        dilate : int ot tuple, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
           Number of blocked connections from input
            channels to output channels, by default 1
        """
        super().__init__(stride, pad, dilate=dilate, group=group)
        self._attrs["op"] = "conv3d_bias"

    def __call__(self, x: Tensor, w: Tensor, b: Tensor) -> List[Tensor]:
        """Call conv3d_bias with tensors x, w, b

        Parameters
        ----------
        x : Tensor
            in shape (N, D, H, W, C_in)
        w : Tensor
            in shape (C_out, K_d, K_h, K_w, C_in)
        b : Tensor
            in shape (C_out)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, D_out, H_out, W_out, C_out)
        """
        self._attrs["inputs"] = [x, w, b]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output
