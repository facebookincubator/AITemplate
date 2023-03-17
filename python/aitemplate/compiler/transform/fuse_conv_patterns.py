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
from aitemplate.compiler.ops.common import elementwise
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.ops.conv import (
    conv2d,
    conv2d_bias,
    conv2d_bias_add,
    conv2d_bias_add_relu,
    conv2d_bias_few_channels,
    conv2d_bias_relu,
    conv2d_bias_relu_few_channels,
    conv2d_bias_sigmoid,
    transposed_conv2d,
    transposed_conv2d_bias,
    transposed_conv2d_bias_relu,
)


def get_conv2d_bias_pattern():
    # Attribute in conv2d is not of concern, it will be passed-through directly.
    return [((conv2d(stride=1, pad=0), elementwise(FuncEnum.ADD)), conv2d_bias)]


def get_conv2d_bias_elementwise_patterns():
    """
    We create the pattern of fusion here.
    The format should be in the form of (pattern, replacement)

    pattern: This would be a list of operator which are chained which we
             want to match
    replacement: The op to replace pattern.
    """

    conv2d_bias_patterns = [
        (
            (
                conv2d_bias(stride=1, pad=0),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.RELU),
            ),
            conv2d_bias_add_relu,
        ),
        (
            (
                conv2d_bias(stride=1, pad=0),
                elementwise(FuncEnum.RELU),
            ),
            conv2d_bias_relu,
        ),
        (
            (
                conv2d_bias(stride=1, pad=0),
                elementwise(FuncEnum.SIGMOID),
            ),
            conv2d_bias_sigmoid,
        ),
        (
            (
                conv2d_bias(stride=1, pad=0),
                elementwise(FuncEnum.ADD),
            ),
            conv2d_bias_add,
        ),
    ]

    transposed_conv2d_bias_patterns = [
        (
            (
                transposed_conv2d_bias(stride=1, pad=0),
                elementwise(FuncEnum.RELU),
            ),
            transposed_conv2d_bias_relu,
        ),
    ]

    transposed_conv2d_patterns = [
        (
            (
                transposed_conv2d(stride=1, pad=0),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.RELU),
            ),
            transposed_conv2d_bias_relu,
        ),
        (
            (
                transposed_conv2d_bias(stride=1, pad=0),
                elementwise(FuncEnum.RELU),
            ),
            transposed_conv2d_bias_relu,
        ),
    ]

    fusion_patterns = (
        conv2d_bias_patterns
        + transposed_conv2d_bias_patterns
        + transposed_conv2d_patterns
    )

    return fusion_patterns


def get_cuda_only_conv2d_bias_elementwise_patterns():
    conv2d_bias_patterns = [
        (
            (
                conv2d_bias_few_channels(stride=1, pad=0),
                elementwise(FuncEnum.RELU),
            ),
            conv2d_bias_relu_few_channels,
        )
    ]

    transposed_conv2d_patterns = [
        (
            (
                transposed_conv2d(stride=1, pad=0),
                elementwise(FuncEnum.ADD),
            ),
            transposed_conv2d_bias,
        ),
    ]

    return conv2d_bias_patterns + transposed_conv2d_patterns
