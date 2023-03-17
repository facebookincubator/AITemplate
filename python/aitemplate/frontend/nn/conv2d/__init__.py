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
# flake8: noqa
"""
modules for conv2d
"""
from aitemplate.frontend.nn.conv2d.conv2d import Conv2d
from aitemplate.frontend.nn.conv2d.conv2d_bias import Conv2dBias
from aitemplate.frontend.nn.conv2d.conv2d_bias_add_hardswish import (
    Conv2dBiasAddHardswish,
)
from aitemplate.frontend.nn.conv2d.conv2d_bias_add_relu import Conv2dBiasAddRelu
from aitemplate.frontend.nn.conv2d.conv2d_bias_few_channels import Conv2dBiasFewChannels
from aitemplate.frontend.nn.conv2d.conv2d_bias_hardswish import Conv2dBiasHardswish
from aitemplate.frontend.nn.conv2d.conv2d_bias_hardswish_few_channels import (
    Conv2dBiasHardswishFewChannels,
)
from aitemplate.frontend.nn.conv2d.conv2d_bias_relu import Conv2dBiasRelu
from aitemplate.frontend.nn.conv2d.conv2d_bias_relu_few_channels import (
    Conv2dBiasReluFewChannels,
)
from aitemplate.frontend.nn.conv2d.conv2d_bias_sigmoid import Conv2dBiasSigmoid
from aitemplate.frontend.nn.conv2d.conv2d_depthwise import Conv2dDepthwise
from aitemplate.frontend.nn.conv2d.conv2d_depthwise_bias import Conv2dDepthwiseBias
from aitemplate.frontend.nn.conv2d.transposed_conv2d_bias import ConvTranspose2dBias
from aitemplate.frontend.nn.conv2d.transposed_conv2d_bias_relu import (
    ConvTranspose2dBiasRelu,
)
