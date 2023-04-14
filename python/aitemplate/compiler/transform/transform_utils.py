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
Util functions for graph transformations.
"""

import logging
from collections import deque
from typing import Dict, List, Union

from aitemplate.compiler.base import Operator, Tensor

from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.transform.mark_param_tensor import mark_param_tensor
from aitemplate.compiler.transform.name_graph import name_graph
from aitemplate.compiler.transform.remove_unused_ops import remove_unused_ops

from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)


def check_graph_validity(sorted_graph: List[Tensor], raiseError: bool = False) -> bool:
    """
    Check whether all tensor/op in the AIT graph matches.
    If raiseError == False, returns true if graph is a valid graph, else false.
       raiseError == True, assert and raise at point of failure.
    """

    def handleError(msg: str):
        if raiseError:
            _LOGGER.info("check_graph_validity() error! Graph:")
            _LOGGER.info(graph_utils.sorted_graph_debug_str(sorted_graph))
            raise RuntimeError(msg)
        else:
            return False

    valid = True
    visited_tensors = set()
    for tensor in sorted_graph:
        tname = tensor._attrs.get("name", None)
        for op in tensor.src_ops():
            if op is None:
                continue
            oname = op._attrs.get("name", None)
            for input_tensor in op._attrs["inputs"]:
                if not isinstance(input_tensor, Tensor):
                    continue
                iname = input_tensor._attrs.get("name", None)
                if input_tensor not in visited_tensors:
                    valid = handleError(
                        "Input tensor {} not established in graph for op {}".format(
                            iname, oname
                        )
                    )
                if op not in input_tensor._attrs["dst_ops"]:
                    valid = handleError(
                        "Op {} not designated as dst_op for tensor {}".format(
                            oname, iname
                        )
                    )
            if tensor not in op._attrs["outputs"]:
                valid = handleError(
                    "Tensor {} not in outputs for op {}".format(tname, oname)
                )
        visited_tensors.add(tensor)

    visited_tensors = set()
    for tensor in sorted_graph[::-1]:
        tname = tensor._attrs.get("name", None)
        for op in tensor.dst_ops():
            if op is None:
                continue
            oname = op._attrs.get("name", None)
            outputs = op._attrs["outputs"]
            if not isinstance(outputs, list):
                outputs = [outputs]
            for output_tensor in outputs:
                if not isinstance(output_tensor, Tensor):
                    continue
                otname = output_tensor._attrs.get("name", None)
                if output_tensor not in visited_tensors:
                    valid = handleError(
                        "Output tensor {} not established in graph for op {}".format(
                            otname, oname
                        )
                    )
                if op not in output_tensor._attrs["src_ops"]:
                    valid = handleError(
                        "Op {} not designated as src_op for tensor {}".format(
                            oname, otname
                        )
                    )
            if tensor not in op._attrs["inputs"]:
                valid = handleError(
                    "Tensor {} not in inputs for op {}".format(tname, oname)
                )
        visited_tensors.add(tensor)

    return valid


def copy_tensor_attributes(dst, src):
    """
    Copy over all tensor attributes that need to be preserved
    """
    attrs = ["depth", "name", "nop", "is_output"]
    for attr in attrs:
        if attr in src._attrs:
            dst._attrs[attr] = src._attrs[attr]


def copy_src_op_attributes(dst, src):
    """
    Copy over all op attributes that need to be preserved. Inputs/outputs
    of the op are not handled by this function.
    """
    if len(dst.src_ops()) != 1 or len(src.src_ops()) != 1:
        return False
    dst_op = list(dst.src_ops())[0]
    src_op = list(src.src_ops())[0]
    attrs = ["alpha"]
    for attr in attrs:
        if attr in src_op._attrs:
            dst_op._attrs[attr] = src_op._attrs[attr]
    return True


def replace_tensor(old_tensor, new_tensor):
    """
    Replaces all references to the old_tensor with the new_tensor.
    """
    if old_tensor._attrs["is_output"]:
        new_tensor._attrs["is_output"] = True
        new_tensor._attrs["name"] = old_tensor._attrs["name"]
    dst_ops = list(old_tensor._attrs["dst_ops"])
    for op in dst_ops:
        op.replace_input_tensor(old_tensor, new_tensor)
        new_tensor._attrs["dst_ops"].add(op)
        old_tensor._attrs["dst_ops"].remove(op)
    remove_tensor_from_sorted_graph(old_tensor)


def replace_tensor_for_op(target_op: Operator, old_tensor: Tensor, new_tensor: Tensor):
    """
    Replaces all references to the old_tensor with the new_tensor in target_op.
    """

    dst_ops = list(old_tensor._attrs["dst_ops"])
    for op in dst_ops:
        if op is target_op:
            op.replace_input_tensor(old_tensor, new_tensor)
            new_tensor._attrs["dst_ops"].add(op)
            old_tensor._attrs["dst_ops"].remove(op)
            break


def remove_dst_op_from_tensor(
    tensors: Union[Tensor, List[Tensor]], dst_op: Operator
) -> None:
    if isinstance(tensors, Tensor):
        tensors._attrs["dst_ops"].remove(dst_op)
    else:
        for tensor in tensors:
            tensor._attrs["dst_ops"].remove(dst_op)


def remove_single_tensor_op_from_sorted_graph(op: Operator) -> None:
    """
    Removes an op which only has one input and one output.
    Connects the previous op and the next op together.
    """
    # Treat reshape op specially because its shape tensors do not maintain
    # input-output dependency on the reshape op.
    if op._attrs["op"] == "reshape":
        # ensure
        for x in op._attrs["inputs"][1:]:
            assert op not in x._attrs["dst_ops"], (
                f'Invalid: shape tensor {x._attrs["name"]} has reshape op '
                f'{op._attrs["name"]} in its dst_ops'
            )
    else:
        assert len(op._attrs["inputs"]) == 1
    assert len(op._attrs["outputs"]) == 1

    input_tensor = op._attrs["inputs"][0]
    output_tensor = op._attrs["outputs"][0]

    input_tensor._attrs["dst_ops"].discard(op)
    input_tensor._attrs["dst_ops"].update(output_tensor._attrs["dst_ops"])

    for dst_op in output_tensor._attrs["dst_ops"]:
        dst_op.replace_input_tensor(output_tensor, input_tensor)
    if output_tensor._attrs["is_output"]:
        assert not input_tensor._attrs[
            "is_input"
        ], f"{input_tensor._attrs['name']} can not be input and output"
        input_tensor._attrs["is_output"] = True
        input_tensor._attrs["name"] = output_tensor._attrs["name"]
        input_tensor._attrs["shape"] = output_tensor._attrs["shape"]

    remove_tensor_from_sorted_graph(output_tensor)


def remove_view_op_from_sorted_graph(op: Operator) -> None:
    """
    Removes a view op, including reshape, squeeze, unsqueeze and flatten.
    """
    # Treat reshape op specially because it may have multiple inputs.
    if op._attrs["op"] != "reshape":
        remove_single_tensor_op_from_sorted_graph(op)
        return

    input_tensor = op._attrs["inputs"][0]
    output_tensor = op._attrs["outputs"][0]

    input_tensor._attrs["dst_ops"].remove(op)
    input_tensor._attrs["dst_ops"].update(output_tensor._attrs["dst_ops"])
    for dst_op in output_tensor._attrs["dst_ops"]:
        dst_op.replace_input_tensor(output_tensor, input_tensor)
    if output_tensor._attrs["is_output"]:
        input_tensor._attrs["is_output"] = True
        input_tensor._attrs["name"] = output_tensor._attrs["name"]
        input_tensor._attrs["shape"] = output_tensor._attrs["shape"]

    # Now we remove this reshape op from its shape tensors' dst_ops
    # Note that a single shape_tensor may be passed to this reshape op multiple
    # times, so we place all shape_tensors into a set to remove duplicates.
    for shape_tensor in set(op._attrs["inputs"][1:]):
        shape_tensor._attrs["dst_ops"].remove(op)

    remove_tensor_from_sorted_graph(output_tensor)


def remove_tensor_from_sorted_graph(tensor: Tensor) -> None:
    """
    Disconnects the tensor from others so that sanitize_sorted_graph()
    could remove it.
    """
    tensor._attrs["src_ops"] = StableSet()
    tensor._attrs["dst_ops"] = StableSet()
    tensor._attrs["is_input"] = False
    tensor._attrs["is_output"] = False


def sanitize_sorted_graph(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Removes tensors whose src_op and dst_ops are empty.
    Inputs and outputs are always kept in the graph.
    Names unnamed tensors.
    """

    if len(sorted_graph) == 1:
        return sorted_graph

    new_sorted_graph = [
        tensor
        for tensor in sorted_graph
        if (len(tensor._attrs["src_ops"]) > 0)
        or (len(tensor._attrs["dst_ops"]) > 0)
        or tensor._attrs["is_input"]
        or tensor._attrs["is_output"]
    ]
    name_graph(new_sorted_graph)
    mark_param_tensor(new_sorted_graph)
    remove_unused_ops(new_sorted_graph)
    check_graph_validity(new_sorted_graph, raiseError=True)
    return new_sorted_graph


def is_ancestor(op1: Operator, op2: Operator) -> bool:
    """
    Returns whether op1 is an ancestor of op2.
    """

    src_ops = deque([op2])
    visited = set()
    while len(src_ops) > 0:
        src_op = src_ops.popleft()
        if src_op in visited:
            continue
        visited.add(src_op)
        for tensor in src_op._attrs["inputs"]:
            if op1 in tensor._attrs["src_ops"]:
                return True
            src_ops.extend(tensor._attrs["src_ops"])
    return False


def is_parent(op1: Operator, op2: Operator) -> bool:
    """
    Returns whether op1 is a parent of op2.
    """

    for tensor in op2._attrs["inputs"]:
        if op1 in tensor._attrs["src_ops"]:
            return True
    return False


def _can_be_constant_folded(tensor: Tensor, visited: Dict[Tensor, bool]):
    if tensor in visited:
        return visited[tensor]
    if tensor._attrs["data"] is not None:
        visited[tensor] = True
        return

    src_ops = tensor.src_ops()

    # might be graph input
    if len(src_ops) == 0:
        visited[tensor] = False
        return

    # weight may have been pre-transposed by other passes
    for src_op in src_ops:
        for input in src_op._attrs["inputs"]:
            _can_be_constant_folded(input, visited)
            if not visited[input]:
                visited[tensor] = False
                return
    visited[tensor] = True


def can_be_constant_folded(tensors: Union[Tensor, List[Tensor]]):
    """
    Check if a tensor or a list of tensors can be folded as a constant
    """
    visited = {}
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    for t in tensors:
        _can_be_constant_folded(t, visited)
        if not visited[t]:
            return False
    return True
