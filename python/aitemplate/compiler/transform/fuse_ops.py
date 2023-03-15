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
import collections
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.ops.common import elementwise, fused_elementwise
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.ops.groupnorm.groupnorm import group_norm
from aitemplate.compiler.ops.groupnorm.groupnorm_swish import group_norm_swish
from aitemplate.compiler.ops.layernorm import layernorm_sigmoid_mul
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.fuse_utils import transform_simple_fusion_patterns
from aitemplate.compiler.transform.toposort import toposort

# pylint: disable=C0103,W0612


_LOGGER = logging.getLogger(__name__)


class SimpleDisjointSet:
    def __init__(self):
        self.node_to_list_mapping: Dict[Any, List[Any]] = {}

    def add(self, node: Any, dependent_nodes: Set[Any]) -> None:
        if node in self.node_to_list_mapping:
            return

        if dependent_nodes is None or len(dependent_nodes) == 0:
            self.node_to_list_mapping[node] = [node]
            return

        current_list = [
            node  # node should also be considered to decide if a new_list can be added.
        ]
        for dependent in list(dependent_nodes):
            if dependent is None or dependent not in self.node_to_list_mapping:
                continue
            new_list = self.node_to_list_mapping.get(dependent)

            if _detect_cycle(current_list + new_list):
                continue
            current_list.extend(new_list)
            for new_node in new_list:
                self.node_to_list_mapping[new_node] = current_list
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

    _LOGGER.debug(
        f"original op set: {original_ops}, to_be_removed_set: {to_be_removed_set}, final_set: {dependent_ops}",
    )
    return dependent_ops


@dataclass
class FusedElementwiseInfo:
    partitioned_ops: List[Operator]
    inputs: Set[Tensor]
    outputs: Set[Tensor]
    external_inputs: Set[Tensor]
    external_outputs: Set[Tensor]


def _partition_subgraphs(ops: List[Operator]) -> Dict[str, Set[Operator]]:
    """
    Given ops of candidate graph of fused_elementwise op graph and partition
    into subgraph based on output shape, returns dict of
    {output shape: ops to form subgraph based on the shape}
    """
    # Partition graph of elementwise into subgraph based on output shape.
    output_op_map = collections.defaultdict(set)
    for op in ops:
        shapes = []
        # Find output nodes
        for output_tensor in op._attrs["outputs"]:
            if (
                output_tensor._attrs["is_output"]
                or len(output_tensor._attrs["dst_ops"] - ops) > 0
            ):
                shapes.append("_".join(map(str, output_tensor._attrs["shape"])))
        # Find anscestor of output node.
        # Outputs with the same shape should form the same graph
        if shapes:
            key = "|".join(shapes)
            op_set = output_op_map[key]
            for anc_op in ops:
                if transform_utils.is_ancestor(anc_op, op):
                    op_set.add(anc_op)
            op_set.add(op)
    return output_op_map


def _get_inputs_outputs(
    partitioned_ops: Set[Operator], all_ops: Set[Operator]
) -> List[Set[Tensor]]:
    """
    Given ops of a partitioned subgraph based on output shape, and ops of full graph
    to form a complete graph with fused_elementwise op, returns all inputs/outputs of
    the ops and the external input/output of the subgraph, which will serve as input/output
    of fused_elementwise op.
    """
    external_inputs = set()
    external_outputs = set()
    tmp_inputs = set()
    tmp_outputs = set()

    for op in partitioned_ops:
        for input_tensor in op._attrs["inputs"]:
            tmp_inputs.add(input_tensor)
            src_ops = set(input_tensor._attrs["src_ops"])
            if (len(src_ops) == 0 or len(src_ops - all_ops) > 0) and (
                not input_tensor.is_a_const_num()
            ):
                external_inputs.add(input_tensor)
            assert op in input_tensor._attrs["dst_ops"]
        for output_tensor in op._attrs["outputs"]:
            tmp_outputs.add(output_tensor)
            dst_ops = set(output_tensor._attrs["dst_ops"])
            if output_tensor._attrs["is_output"] or len(dst_ops - all_ops) > 0:
                external_outputs.add(output_tensor)
            assert len(output_tensor._attrs["src_ops"]) == 1
            assert list(output_tensor._attrs["src_ops"])[0] == op

    assert (
        external_inputs == tmp_inputs - tmp_outputs
    ), "external_inputs: {} is not equal to tmp_inputs: {} - tmp_outputs: {}.".format(
        external_inputs, tmp_inputs, tmp_outputs
    )
    assert (
        len(tmp_outputs - tmp_inputs - external_outputs) == 0
    ), "tmp_outputs: {} - tmp_inputs: {} - external_outputs: {} is not empty.".format(
        tmp_outputs, tmp_inputs, external_outputs
    )
    assert (
        len(external_outputs - tmp_outputs) == 0
    ), "external_outputs: {} - tmp_outputs: {} is not empty.".format(
        external_outputs, tmp_outputs
    )

    return [tmp_inputs, tmp_outputs, external_inputs, external_outputs]


def _collect_info(
    output_op_map: Dict[str, Set[Operator]],
    all_ops: Set[Operator],
    sorted_graph: List[Tensor],
) -> List[FusedElementwiseInfo]:
    """
    Collects information for each fused_elementwise op:
        1. Provide op_list in topological order so fuse_elementwise backend can emit operations in order.
        2. Provide inputs outputs info of each subgraph. This need to happen before fuse ops are created,
        i.e. graph get changed.
    Returns list of fused_op_info, which contains:
        partitioned op list in topological order, all inputs/outputs of elementwise ops and
        their external input/output, serving as input/output of fused_elementwise op.
    """
    info_list = []
    for op_set in output_op_map.values():
        # Toposort the op_set into op_list
        # because fuse_elementwise stores elementwise ops in topological order
        topo_set = set()
        op_list = []
        for tensor in sorted_graph:
            topo_set.add(tensor)
            to_remove = set()
            for op in op_set:
                if all([arg in topo_set for arg in op._attrs["inputs"]]):
                    op_list.append(op)
                    to_remove.add(op)
            op_set = op_set - to_remove
        assert (
            not op_set
        ), "Unable to find topological order of op list for fused_elementwise!"
        # Get all inputs/outputs of elementwise ops and their external input/output,
        # which will serve as input/output of fused_elementwise op.
        inputs_outputs = _get_inputs_outputs(op_list, all_ops)
        fused_op_info = FusedElementwiseInfo(op_list, *inputs_outputs)
        info_list.append(fused_op_info)
    return info_list


def _create_fuse_ops(info_list: List[FusedElementwiseInfo]) -> None:
    """
    Creates fused ops based on info we collected.
    First is to update elementwise ops' inputs/outputs within the subgraph;
    Second is to create fused_elementwise ops where their inputs/outputs
    are external inputs/outputs of the subgraph.
    """
    for info in info_list:
        op_set = set(info.partitioned_ops)
        for tensor in info.inputs | info.outputs:
            tensor._attrs["src_ops"] = tensor._attrs["src_ops"] - op_set
            tensor._attrs["dst_ops"] = tensor._attrs["dst_ops"] - op_set
        fused_elementwise(
            info.partitioned_ops,
            info.external_inputs,
            info.external_outputs,
        )


def _detect_cycle(group: List[Operator]) -> bool:
    """
    Given a group of ops, to detect if they would form cycles, i.e.
      --> group_ops
     /      /
    A <-----
    we need to find all parents of all ops in that group
    and see if any parent's ancester (execluding the ones already in the group) exists in the group.
    """
    parents = [o for op1 in group for i in op1._attrs["inputs"] for o in i.src_ops()]
    for op1 in group:
        for op2 in set(parents) - set(group):
            if transform_utils.is_ancestor(op1, op2):
                return True
    return False


def fuse_elementwise(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    """
    Given a sorted graph, returns a sorted graph with fused_elementwise ops on fusable elementwise ops.
    """
    disjoint_set = SimpleDisjointSet()
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] == "elementwise":
            disjoint_set.add(
                src_op,
                _find_fusable_elementwise_ops(src_op),
            )

    to_be_fused_op_groups = disjoint_set.get_node_groups()

    for ops in to_be_fused_op_groups:
        # Partition subgraph based on output shape.
        output_op_map = _partition_subgraphs(ops)
        # Collect information to create fuse ops.
        info_list = _collect_info(output_op_map, set(ops), sorted_graph)
        # Create fuse ops.
        _create_fuse_ops(info_list)

    sorted_graph = toposort(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def process_singleton_elementwise(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """
    A dummy pass which enables codegen for any elementwise op without fusing it with neighbors
    """
    disjoint_set = SimpleDisjointSet()
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] == "elementwise":
            disjoint_set.add(
                src_op,
                {src_op},
            )

    to_be_fused_op_groups = disjoint_set.get_node_groups()

    for ops in to_be_fused_op_groups:
        # Partition subgraph based on output shape.
        # output_op_map = {op._attrs["op"]: set(op) for op in ops}
        output_op_map = _partition_subgraphs(ops)
        # Collect information to create fuse ops.
        info_list = _collect_info(output_op_map, set(ops), sorted_graph)
        # Create fuse ops.
        _create_fuse_ops(info_list)

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


def _fuse_groupnorm_sigmoid_mul(sorted_graph: List[Tensor]) -> List[Tensor]:
    fusion_patterns = [
        (
            (
                group_norm(num_groups=2, num_channels=4),
                elementwise(FuncEnum.SIGMOID),
                elementwise(FuncEnum.MUL),
            ),
            group_norm_swish,
        )
    ]
    graph = transform_simple_fusion_patterns(sorted_graph, fusion_patterns)
    return graph


def fuse_ops(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    funcs = [
        _fuse_layernorm_sigmoid_mul,
        _fuse_groupnorm_sigmoid_mul,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
