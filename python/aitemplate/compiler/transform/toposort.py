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
Graph pass for topological sort.
"""
from typing import List, Union

from aitemplate.compiler.base import Tensor

# pylint: disable=C0103


def toposort(nodes: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    """Generate sorted nodes by topological order. This is the foundation of all graph passes.

    Parameters
    ----------
    nodes : Union[Tensor, List[Tensor]]
        The output of the model

    Returns
    -------
    List[Tensor]
        Sorted graph
    """
    visited = set()
    sorted_graph = []
    stack = []

    if isinstance(nodes, Tensor):
        stack.append((nodes, False))
    else:
        for node in list(nodes)[::-1]:
            stack.append((node, False))

    while len(stack) > 0:
        curr_node, curr_visited = stack.pop()
        if curr_visited:
            sorted_graph.append(curr_node)
            for src_op in curr_node.src_ops():
                for next_node in src_op._attrs["outputs"]:
                    stack.append((next_node, False))
            continue
        if curr_node in visited:
            continue

        visited.add(curr_node)
        stack.append((curr_node, True))
        for src_op in curr_node.src_ops():
            args = src_op._attrs["inputs"]
            indexed_args = list(enumerate(args))
            depth_first_args = sorted(
                indexed_args, key=lambda x: x[1]._attrs["depth"], reverse=True
            )
            visit_seq = [x[0] for x in depth_first_args[::-1]]
            for idx in visit_seq:
                arg = args[idx]
                stack.append((arg, False))

    return sorted_graph
