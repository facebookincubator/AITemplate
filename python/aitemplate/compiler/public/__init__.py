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
This file defines public tensor concepts and ops that are exposed to
external converters, e.g. FX2AITemplate.
"""

# pylint: disable=C0413,W0105

"""Shape"""
from aitemplate.compiler.base import IntImm, IntVar

"""Tensor"""
from aitemplate.compiler.base import IntVarTensor, Tensor

"""Profiling"""
from aitemplate.compiler.base import DynamicProfileStrategy

"""Operators"""

"""Elementwise"""
from aitemplate.compiler.ops.common.elementwise import clamp, elementwise
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from aitemplate.compiler.ops.common.int_elementwise import int_elementwise

"""GEMM"""
from aitemplate.compiler.ops.gemm_universal.bmm_xxx import bmm_rcr, bmm_rrr
from aitemplate.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias import gemm_rcr_bias
from aitemplate.compiler.ops.gemm_universal.gemm_rrr import gemm_rrr

"""Reduce"""
from aitemplate.compiler.ops.reduce.reduce_mean import reduce_mean
from aitemplate.compiler.ops.reduce.reduce_sum import reduce_sum
from aitemplate.compiler.ops.reduce.var import var
from aitemplate.compiler.ops.reduce.vector_norm import vector_norm

"""View ops"""
from aitemplate.compiler.ops.common.view_ops import flatten, reshape, squeeze, unsqueeze

"""Functions"""
from aitemplate.compiler.ops.conv.conv2d import conv2d
from aitemplate.compiler.ops.conv.conv2d_bias import conv2d_bias
from aitemplate.compiler.ops.conv.conv2d_bias_relu import conv2d_bias_relu
from aitemplate.compiler.ops.conv.conv3d import conv3d
from aitemplate.compiler.ops.conv.conv3d_bias import conv3d_bias
from aitemplate.compiler.ops.conv.depthwise_conv3d import depthwise_conv3d
from aitemplate.compiler.ops.conv.transposed_conv2d import transposed_conv2d
from aitemplate.compiler.ops.conv.transposed_conv2d_bias import transposed_conv2d_bias
from aitemplate.compiler.ops.groupnorm.groupnorm import group_norm
from aitemplate.compiler.ops.layernorm.group_layernorm import group_layernorm
from aitemplate.compiler.ops.layernorm.group_layernorm_sigmoid_mul import (
    group_layernorm_sigmoid_mul,
)
from aitemplate.compiler.ops.layernorm.layernorm import layernorm
from aitemplate.compiler.ops.padding import ndhwc3to8, nhwc3to8, pad_last_dim
from aitemplate.compiler.ops.pool.avg_pool2d import avg_pool2d
from aitemplate.compiler.ops.pool.max_pool2d import max_pool2d
from aitemplate.compiler.ops.softmax.softmax import softmax
from aitemplate.compiler.ops.tensor.size import size
from aitemplate.compiler.ops.tensor.topk import topk

"""Memory ops"""
from aitemplate.compiler.ops.tensor.chunk import chunk
from aitemplate.compiler.ops.tensor.concatenate import concatenate
from aitemplate.compiler.ops.tensor.dynamic_slice import dynamic_slice
from aitemplate.compiler.ops.tensor.expand import expand
from aitemplate.compiler.ops.tensor.full import full
from aitemplate.compiler.ops.tensor.permute import permute
from aitemplate.compiler.ops.tensor.split import split

"""Python ops"""
from aitemplate.compiler.ops.common import getitem, list_construct, tuple_construct
