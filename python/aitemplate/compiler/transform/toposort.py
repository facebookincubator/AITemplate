# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from typing import List, Union

from ..base import Tensor

# pylint: disable=C0103


def toposort(nodes: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    """[summary]

    Parameters
    ----------
    nodes : Tensor or List[Tensor]
        [description]

    Returns
    -------
    List[Tensor]
        [description]
    """
    visited = set()
    sorted_graph = []

    def DFS(nd: Tensor):
        if nd in visited:
            return
        for src_op in nd.src_ops():
            args = src_op._attrs["inputs"]
            indexed_args = list(enumerate(args))
            depth_first_args = sorted(
                indexed_args, key=lambda x: x[1]._attrs["depth"], reverse=True
            )
            visit_seq = [x[0] for x in depth_first_args]
            for idx in visit_seq:
                arg = args[idx]
                DFS(arg)
        visited.add(nd)
        sorted_graph.append(nd)
        for src_op in nd.src_ops():
            for next_nd in src_op._attrs["outputs"]:
                DFS(next_nd)

    if isinstance(nodes, Tensor):
        DFS(nodes)
    else:
        for node in list(nodes):
            DFS(node)
    return sorted_graph
