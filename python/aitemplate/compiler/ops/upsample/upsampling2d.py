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
Upsampling2d op.
"""
from aitemplate.compiler.ops.upsample.upsampling_common import upsampling2d_base


# pylint: disable=C0103
class upsampling2d(upsampling2d_base):
    """
    Applies a 2D bilinear upsampling to an input signal composed of several input
    channels.

    To specify the scale, it takes the :attr:`scale_factor` as it's constructor argument.

    * :attr:`scale_factor` (float): multiplier for spatial size.

    Args:
        input (Tensor [N, H, W, C]): the input data.

    Return:
        Tensor [N, H_out, W_out, C].
    """

    def __init__(self, scale_factor, mode) -> None:
        super().__init__(scale_factor, mode)
        self._attrs["op"] = "upsampling2d"
        self._attrs["mode"] = mode
