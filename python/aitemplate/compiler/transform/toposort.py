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
import heapq
from typing import List, Tuple, Union

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
    return _priSort(nodes, SizePriTensorHelper())


def _dfsSort(nodes: Union[Tensor, List[Tensor]]) -> List[Tensor]:
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


class PriTensorHelper:
    def __init__(self) -> None:
        self.entry_cnt = -1

    def get_heap_input(self, node: Tensor) -> Tuple[float, int, Tensor]:
        # input is built based on heapq doc suggestion:
        # https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
        # the return tuple is: (
        #   priority_ (less is more important),
        #   entry_cnt (so earlier entered item is chosen if same priority),
        #   element (here is tensor)
        # )
        self.entry_cnt += 1
        return (
            self.get_priority(node),
            self.entry_cnt,
            node,
        )

    def get_tensor_from_heap_output(
        self, heap_output: Tuple[float, int, Tensor]
    ) -> Tensor:
        return heap_output[2]

    def get_priority(self, node: Tensor) -> float:
        # please implement your own priority function
        # note that smaller value would be in higher-pri
        pass


class SizePriTensorHelper(PriTensorHelper):
    def get_priority(self, node: Tensor) -> float:
        # use negative byte size since
        # we'd like to pop larger size first
        return -node.size_bytes()


def _priSort(
    nodes: Union[Tensor, List[Tensor]], pri_tensor_helper: PriTensorHelper
) -> List[Tensor]:
    # do a DFS to get all nodes in a list
    nodes = _dfsSort(nodes)
    # number of src tensors
    in_degree = {}
    for node in nodes:
        in_degree[node] = 0
        for src_op in node.src_ops():
            # sometimes it'd have 2 same nodes in one list
            # change to set to de-dupe these nodes
            in_degree[node] += len(set(src_op._attrs["inputs"]))

    queue = []
    sorted_graph = []
    for node in nodes:
        if in_degree[node] == 0:
            # input nodes need to be in the original order,
            # hence add them to the sorted graph here
            # instead of going through the pri heap
            sorted_graph.append(node)
            heapq.heappush(queue, pri_tensor_helper.get_heap_input(node))

    while queue:
        node = pri_tensor_helper.get_tensor_from_heap_output(heapq.heappop(queue))
        if node not in sorted_graph:
            sorted_graph.append(node)

        for dst_op in node.dst_ops():
            for next_node in set(dst_op._attrs["outputs"]):
                if next_node not in in_degree:
                    continue
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0:
                    heapq.heappush(queue, pri_tensor_helper.get_heap_input(next_node))
    return sorted_graph
