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
Fuse conv + elementwise ops.
"""
from typing import List

from aitemplate.compiler.base import Tensor

from aitemplate.compiler.transform.fuse_conv_patterns import (
    get_conv2d_bias_elementwise_patterns,
    get_conv2d_bias_pattern,
    get_cuda_only_conv2d_bias_elementwise_patterns,
)
from aitemplate.compiler.transform.fuse_utils import transform_simple_fusion_patterns

# pylint: disable=C0103,C0415,W0612


def _transform_conv2d_bias(sorted_graph: List[Tensor]) -> List[Tensor]:
    fusion_patterns = get_conv2d_bias_pattern()

    return transform_simple_fusion_patterns(sorted_graph, fusion_patterns)


def _transform_conv2d_bias_elementwise(sorted_graph: List[Tensor]) -> List[Tensor]:
    fusion_patterns = get_conv2d_bias_elementwise_patterns()

    return transform_simple_fusion_patterns(sorted_graph, fusion_patterns)


def _transform_cuda_only_conv2d_bias_elementwise(
    sorted_graph: List[Tensor],
) -> List[Tensor]:
    fusion_patterns = get_cuda_only_conv2d_bias_elementwise_patterns()

    return transform_simple_fusion_patterns(sorted_graph, fusion_patterns)


def fuse_conv_elementwise(sorted_graph: List[Tensor], _: str) -> List[Tensor]:
    """
    Fuse conv + elementwise ops. The second argument is unused, it's only
    here to make the type of this function the same as the others called in optimize_graph.
    """
    funcs = [
        _transform_conv2d_bias,
        _transform_conv2d_bias_elementwise,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)

    from aitemplate.backend.target import Target

    if Target.current().name() == "cuda":
        funcs = [
            _transform_cuda_only_conv2d_bias_elementwise,
        ]
        for func in funcs:
            sorted_graph = func(sorted_graph)
    return sorted_graph
