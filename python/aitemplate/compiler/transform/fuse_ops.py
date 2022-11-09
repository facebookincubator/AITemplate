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
Perform operator fusions.
"""
from typing import Any, Dict, List, Set

from aitemplate.compiler.base import Operator
from aitemplate.compiler.transform.toposort import toposort

from aitemplate.utils import logger

from ..base import Tensor
from ..ops.common import fused_elementwise
from ..ops.common.epilogue import FuncEnum
from ..ops.layernorm import layernorm_sigmoid_mul
from . import transform_utils

# pylint: disable=C0103,W0612


class SimpleDisjointSet(object):
    def __init__(self):
        self.node_to_list_mapping: Dict[Any, List[Any]] = {}

    def add(self, node: Any, dependent_nodes: Set[Any]) -> None:
        if node in self.node_to_list_mapping:
            return

        if dependent_nodes is None or len(dependent_nodes) == 0:
            self.node_to_list_mapping[node] = [node]
            return

        current_list = None
        for dependent in dependent_nodes:
            if dependent is None or dependent not in self.node_to_list_mapping:
                continue
            new_list = self.node_to_list_mapping.get(dependent)
            if current_list is None:
                current_list = new_list
            elif current_list is not new_list:
                current_list.extend(new_list)
                for new_node in new_list:
                    self.node_to_list_mapping[new_node] = current_list
        if current_list is None:
            current_list = []
        current_list.append(node)
        self.node_to_list_mapping[node] = current_list

    def get_node_groups(self) -> List[List[Any]]:
        node_groups = []
        visited = set()
        for groups in self.node_to_list_mapping.values():
            addr = id(groups)
            if addr not in visited:
                visited.add(addr)
                node_groups.append(groups)
        return node_groups


def _find_fusable_elementwise_ops(op: Operator) -> Set[Operator]:
    """
    Given an elementwise op, returns a list of parent elementwise ops
    which can be fused with this elementwise op.
    """

    # Get parent ops.
    dependent_ops = set()
    for input_tensor in op._attrs["inputs"]:
        dependent_ops.update(input_tensor._attrs["src_ops"])
    original_ops = set(dependent_ops)

    # First, filter out all non-elementwise ops.
    to_be_removed_set = set()
    for op in dependent_ops:
        if op._attrs["op"] != "elementwise":
            to_be_removed_set.add(op)
        else:
            # Assuming there are two elementwise ops, op1 and op2, where op1 is a
            # parent op of op2. If op1's output is an output tensor, or if op1 is
            # consumed by other non-elementwise ops, op1 cannot be fused with op2.
            output = op._attrs["outputs"][0]
            if output._attrs["is_output"]:
                to_be_removed_set.add(op)
                continue
            for next_op in output.dst_ops():
                if next_op._attrs["op"] != "elementwise":
                    to_be_removed_set.add(op)

    dependent_ops = dependent_ops - to_be_removed_set

    # Then get all connected elementwise ops at the last layer.
    while True:
        for op1 in dependent_ops:
            # If op1 is an ancestor of op2 but not a parent of op2,
            # op1 and op2 cannot be fused. Remove op1 and only
            # keep op2.
            for op2 in dependent_ops:
                if op1 is op2:
                    continue
                if transform_utils.is_ancestor(
                    op1, op2
                ) and not transform_utils.is_parent(op1, op2):
                    to_be_removed_set.add(op1)

            # If op1 is an ancestor of a removed op,
            # op1 and op cannot be fused. Remove op1.
            for op2 in list(to_be_removed_set):
                if transform_utils.is_ancestor(op1, op2):
                    to_be_removed_set.add(op1)

        prev_len = len(dependent_ops)
        dependent_ops = dependent_ops - to_be_removed_set
        new_len = len(dependent_ops)
        if prev_len == new_len:
            break

    logger.debug(
        __file__,
        f"original op set: {original_ops}, to_be_removed_set: {to_be_removed_set}, final_set: {dependent_ops}",
    )
    return dependent_ops


def _fuse_elementwise(sorted_graph: List[Tensor]) -> List[Tensor]:
    disjoint_set = SimpleDisjointSet()
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] == "elementwise":
            disjoint_set.add(src_op, _find_fusable_elementwise_ops(src_op))

    to_be_fused_op_groups = disjoint_set.get_node_groups()
    for ops in to_be_fused_op_groups:
        fused_elementwise(ops)

    sorted_graph = toposort(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _fuse_layernorm_sigmoid_mul(sorted_graph: List[Tensor]) -> List[Tensor]:
    to_be_fused_op_groups = []
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op is None:
            continue
        if src_op._attrs["op"] != "layernorm":
            continue
        layer_norm = src_op

        dst_ops = list(tensor._attrs["dst_ops"])
        if not dst_ops:
            continue

        # layernorm as the last op in the graph
        next_op = dst_ops[0]
        if (
            next_op._attrs["op"] != "elementwise"
            or next_op._attrs["func"] != FuncEnum.SIGMOID
        ):
            continue
        sigmoid = next_op

        next_tensor = sigmoid._attrs["outputs"][0]

        # layernorm + sigmoid
        dst_ops = list(next_tensor._attrs["dst_ops"])
        if not dst_ops:
            continue

        next_op = dst_ops[0]
        if (
            next_op._attrs["op"] != "elementwise"
            or next_op._attrs["func"] != FuncEnum.MUL
        ):
            continue
        mul = next_op

        if layernorm_sigmoid_mul.is_valid(layer_norm, sigmoid, mul):
            to_be_fused_op_groups.append((layer_norm, sigmoid, mul))

    for ops in to_be_fused_op_groups:
        layernorm_sigmoid_mul(*ops)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def fuse_ops(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    funcs = [
        _fuse_layernorm_sigmoid_mul,
        _fuse_elementwise,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
