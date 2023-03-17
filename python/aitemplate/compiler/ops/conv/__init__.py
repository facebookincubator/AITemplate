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
Conv2d family operators.
"""
from aitemplate.compiler.ops.conv.conv2d import conv2d
from aitemplate.compiler.ops.conv.conv2d_bias import conv2d_bias
from aitemplate.compiler.ops.conv.conv2d_bias_add import conv2d_bias_add
from aitemplate.compiler.ops.conv.conv2d_bias_add_hardswish import (
    conv2d_bias_add_hardswish,
)
from aitemplate.compiler.ops.conv.conv2d_bias_add_relu import conv2d_bias_add_relu
from aitemplate.compiler.ops.conv.conv2d_bias_few_channels import (
    conv2d_bias_few_channels,
)
from aitemplate.compiler.ops.conv.conv2d_bias_hardswish import conv2d_bias_hardswish
from aitemplate.compiler.ops.conv.conv2d_bias_hardswish_few_channels import (
    conv2d_bias_hardswish_few_channels,
)
from aitemplate.compiler.ops.conv.conv2d_bias_relu import conv2d_bias_relu
from aitemplate.compiler.ops.conv.conv2d_bias_relu_few_channels import (
    conv2d_bias_relu_few_channels,
)
from aitemplate.compiler.ops.conv.conv2d_bias_sigmoid import conv2d_bias_sigmoid
from aitemplate.compiler.ops.conv.conv2d_depthwise import conv2d_depthwise
from aitemplate.compiler.ops.conv.conv2d_depthwise_bias import conv2d_depthwise_bias
from aitemplate.compiler.ops.conv.conv3d import conv3d
from aitemplate.compiler.ops.conv.conv3d_bias import conv3d_bias
from aitemplate.compiler.ops.conv.depthwise_conv3d import depthwise_conv3d
from aitemplate.compiler.ops.conv.transposed_conv2d import transposed_conv2d
from aitemplate.compiler.ops.conv.transposed_conv2d_bias import transposed_conv2d_bias
from aitemplate.compiler.ops.conv.transposed_conv2d_bias_relu import (
    transposed_conv2d_bias_relu,
)
