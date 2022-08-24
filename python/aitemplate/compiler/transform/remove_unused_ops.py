# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Remove useless operators from a sorted_graph.
"""
from collections import deque
from typing import List

from ..base import Tensor


def remove_unused_ops(sorted_graph: List[Tensor]) -> None:
    """Remove ops which are not src operators of tensors in the input sorted_graph."""

    src_ops = set()
    to_be_visited_ops = deque()
    for node in sorted_graph:
        src_ops.update(node._attrs["src_ops"])
        to_be_visited_ops.extend(node._attrs["dst_ops"])

    visited_ops = set()
    while len(to_be_visited_ops) > 0:
        next_op = to_be_visited_ops.popleft()
        if next_op in visited_ops:
            continue
        visited_ops.add(next_op)
        if next_op not in src_ops:
            for input_tensor in next_op._attrs["inputs"]:
                input_tensor._attrs["dst_ops"].discard(next_op)
        for output_tensor in next_op._attrs["outputs"]:
            to_be_visited_ops.extend(output_tensor._attrs["dst_ops"])
