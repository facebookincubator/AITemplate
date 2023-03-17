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
Upsampling2d_add op.
"""
from typing import List

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.upsample.upsampling_common import upsampling2d_base


# pylint: disable=C0103
class upsampling2d_add(upsampling2d_base):
    """
    Fused op for bilinear_upsampling + add.

    Applies a 2D bilinear upsampling to an input signal composed of several input
    channels, and adds an residual.

    To specify the scale, it takes the :attr:`scale_factor` as it's constructor argument.

    * :attr:`scale_factor` (float): multiplier for spatial size.

    Args:
        input (Tensor [N, H, W, C]): the input data.
        r (Tensor [N, H_out, W_out, C]): the residual.

    Return:
        Tensor [N, H_out, W_out, C].
    """

    def __init__(self, scale_factor, mode) -> None:
        super().__init__(scale_factor, mode)
        self._attrs["op"] = "upsampling2d_add"
        self._attrs["mode"] = mode

    def __call__(self, x: Tensor, r: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x, r]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output
