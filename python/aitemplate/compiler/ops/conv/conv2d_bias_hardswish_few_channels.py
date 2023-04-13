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
Fused conv2d_bias_hardswish_few_channels op.
"""
from aitemplate.compiler.ops.conv.special_conv2d_bias_activation import (
    special_conv2d_bias_activation,
)


# pylint: disable=C0103
class conv2d_bias_hardswish_few_channels(special_conv2d_bias_activation):
    """conv2d_bias_hardswish_few_channels.

    This operator equals to conv2d_bias_hardswish but has imporved performance for in_channels < 8.
    """

    def __init__(self, stride, pad, dilate=1, auto_padding=True) -> None:
        """Initializes conv2d_bias_relu_few_channels"""
        super().__init__("hardswish", stride, pad, dilate, auto_padding)

    def _get_op_attributes(self):
        attr = super()._get_op_attributes()
        del attr["activation"]

        return attr
