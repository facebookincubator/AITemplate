# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Applies graph transformations.
"""

from typing import List

from ...utils import graph_utils
from ..base import Tensor
from .apply_padding import apply_padding
from .fuse_mm_elementwise import fuse_mm_elementwise
from .fuse_ops import fuse_ops
from .fuse_parallel_gemms import fuse_parallel_gemms
from .fuse_permute_bmm import fuse_permute_bmm
from .transform_memory_ops import transform_memory_ops
from .transform_odd_alignment import transform_odd_alignment
from .transform_special_ops import transform_special_ops
from .transform_strided_ops import transform_strided_ops


def optimize_graph(sorted_graph: List[Tensor], workdir: str) -> List[Tensor]:
    """
    Applies graph optimizations.
    """

    funcs = [
        fuse_permute_bmm,
        transform_odd_alignment,
        fuse_mm_elementwise,
        transform_memory_ops,
        fuse_ops,
        # need to run before transform_strided_ops to fuse strided ops + concat
        # and transform_memory_ops to fuse split + concat
        fuse_parallel_gemms,
        # This needs to be run after fuse_ops() to avoid handling elementwise
        # op directly. After fuse_ops, there are only FusedElementwise ops.
        transform_special_ops,
        apply_padding,
        transform_strided_ops,
        transform_memory_ops,
    ]

    for func in funcs:
        sorted_graph = func(sorted_graph)
        graph_utils.dump_graph_debug_str_to_file(sorted_graph, workdir, func.__name__)

    return sorted_graph
