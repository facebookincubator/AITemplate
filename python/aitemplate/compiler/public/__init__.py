# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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

"""GeMM"""
from aitemplate.compiler.ops.gemm_universal.bmm_rcr import bmm_rcr
from aitemplate.compiler.ops.gemm_universal.bmm_rrr import bmm_rrr
from aitemplate.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr
from aitemplate.compiler.ops.gemm_universal.gemm_rrr import gemm_rrr

"""Reduce"""
from aitemplate.compiler.ops.reduce.reduce_mean import reduce_mean
from aitemplate.compiler.ops.reduce.reduce_sum import reduce_sum
from aitemplate.compiler.ops.reduce.var import var
from aitemplate.compiler.ops.reduce.vector_norm import vector_norm

"""View ops"""
from aitemplate.compiler.ops.common.view_ops import flatten, reshape, squeeze, unsqueeze

"""Functions"""
from aitemplate.compiler.ops.layernorm.group_layernorm import group_layernorm
from aitemplate.compiler.ops.layernorm.group_layernorm_sigmoid_mul import (
    group_layernorm_sigmoid_mul,
)
from aitemplate.compiler.ops.layernorm.layernorm import layernorm
from aitemplate.compiler.ops.softmax.softmax import softmax
from aitemplate.compiler.ops.tensor.size import size
from aitemplate.compiler.ops.tensor.topk import topk

"""Memory ops"""
from aitemplate.compiler.ops.tensor.chunk import chunk
from aitemplate.compiler.ops.tensor.concatenate import concatenate
from aitemplate.compiler.ops.tensor.dynamic_slice import dynamic_slice
from aitemplate.compiler.ops.tensor.permute import permute
from aitemplate.compiler.ops.tensor.split import split

"""PyThon ops"""
from aitemplate.compiler.ops.common import getitem, list_construct, tuple_construct
