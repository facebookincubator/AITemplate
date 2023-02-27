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
Applies graph transformations.
"""

from typing import List

from ...utils import graph_utils
from ..base import Tensor
from .apply_padding import apply_padding
from .fuse_conv_elementwise import fuse_conv_elementwise
from .fuse_group_ops import fuse_group_ops
from .fuse_mm_elementwise import fuse_mm_elementwise
from .fuse_mm_reshape_permute import fuse_mm_reshape_permute
from .fuse_ops import fuse_ops
from .fuse_parallel_gemms import fuse_parallel_gemms
from .fuse_permute_bmm_and_gemm import fuse_permute_bmm_and_gemm
from .split_large_concat_ops import split_large_concat_ops
from .split_large_slice_scatter_ops import split_large_slice_scatter_ops
from .split_large_split_ops import split_large_split_ops
from .transform_memory_ops import transform_memory_ops
from .transform_odd_alignment import transform_odd_alignment
from .transform_special_ops import transform_special_ops
from .transform_strided_ops import transform_strided_ops


def optimize_graph(sorted_graph: List[Tensor], workdir: str) -> List[Tensor]:
    """Applies graph optimizations, including

    - fuse permute and bmm
    - fuse permute and gemm
    - transform odd alignment
    - fuse conv and elementwise
    - fuse gemm and elementwise
    - fuse elementwise ops
    - fuse parallel gemms
    - fuse group ops
    - transform special ops
    - transform strided ops
    - transform memory ops
    - apply padding

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph
    workdir : str
        working directory

    Returns
    -------
    List[Tensor]
        Fused graph
    """

    funcs = [
        fuse_permute_bmm_and_gemm,
        transform_odd_alignment,
        fuse_conv_elementwise,
        fuse_mm_elementwise,
        fuse_mm_reshape_permute,
        transform_memory_ops,
        fuse_ops,
        # need to run before transform_strided_ops to fuse strided ops + concat
        # and transform_memory_ops to fuse split + concat
        fuse_parallel_gemms,
        fuse_group_ops,
        # This needs to be run after fuse_ops() to avoid handling elementwise
        # op directly. After fuse_ops, there are only FusedElementwise ops.
        transform_special_ops,
        apply_padding,
        transform_strided_ops,
        split_large_slice_scatter_ops,
        split_large_concat_ops,
        split_large_split_ops,
        transform_memory_ops,
    ]

    for func in funcs:
        sorted_graph = func(sorted_graph, workdir)
        graph_utils.dump_graph_debug_str_to_file(sorted_graph, workdir, func.__name__)

    return sorted_graph
