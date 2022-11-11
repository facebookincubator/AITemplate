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
from .bind_constants import bind_constants
from .constant_folding import constant_folding
from .fuse_conv_elementwise import fuse_conv_elementwise
from .fuse_group_ops import (
    fuse_group_gemm_ops,
    fuse_group_layernorm_ops,
    fuse_group_ops,
)
from .fuse_mm_elementwise import fuse_mm_elementwise
from .fuse_ops import fuse_ops
from .fuse_permute_bmm_and_gemm import fuse_permute_bmm_and_gemm
from .mark_param_tensor import mark_param_tensor, mark_special_views
from .memory_planning import memory_planning
from .name_graph import name_graph
from .optimize_graph import optimize_graph
from .profile import profile
from .refine_graph import refine_graph
from .remove_no_ops import remove_no_ops
from .remove_unused_ops import remove_unused_ops
from .split_large_concat_ops import split_large_concat_ops
from .toposort import toposort
from .transform_memory_ops import transform_memory_ops
from .transform_odd_alignment import transform_odd_alignment
from .transform_special_ops import transform_special_ops
from .transform_strided_ops import transform_strided_ops
