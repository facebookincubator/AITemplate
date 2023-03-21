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
from typing import Any, List, Optional, Set

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.compiler.transform.transform_utils import (
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    replace_tensor,
    sanitize_sorted_graph,
)

# pylint: disable=C0103,C0415,W0612


def extract_only_one_op(ops: Set[Optional[Operator]]) -> Optional[Operator]:
    """
    Helper function that returns the op from src_ops() or dst_ops() call.
    Return None if there are no ops or if there's more than one op.
    """
    if ops is None or len(ops) != 1:
        return None
    return list(ops)[0]


def is_elementwise_type(op: Operator, elementwise_type):
    if op._attrs["op"] != "elementwise":
        return False
    return op._attrs["func"] == elementwise_type


def _is_same_op_type(op_A: Operator, op_B: Operator):
    """
    Compare whether 2 ops are of same type.
    """
    if op_A._attrs["op"] != op_B._attrs["op"]:
        return False
    if op_A._attrs["op"] == "elementwise":
        if op_A._attrs["func"] != op_B._attrs["func"]:
            return False

    return True


def _find_fusion_root(tensor: Tensor, fusion_patterns: List[Any]) -> int:
    fusion_idx = -1

    src_op = extract_only_one_op(tensor._attrs["src_ops"])
    if src_op is None:
        return fusion_idx

    for idx, fusion_pattern in enumerate(fusion_patterns):
        pattern, _ = fusion_pattern
        curr_op = src_op
        curr_tensor = tensor

        for step, pattern_op in enumerate(pattern):
            if not _is_same_op_type(curr_op, pattern_op):
                break
            check_input = getattr(pattern_op, "is_valid_inputs", None)
            if check_input is not None:
                valid, _ = check_input(*curr_op._attrs["inputs"])
                if not valid:
                    break

            if step == len(pattern) - 1:
                fusion_idx = idx
                break

            dst_op = extract_only_one_op(curr_tensor._attrs["dst_ops"])
            if dst_op is None:
                break
            curr_op = dst_op
            dst_op_tensor = dst_op._attrs["outputs"]
            if len(dst_op_tensor) != 1:
                break
            curr_tensor = dst_op_tensor[0]

        if fusion_idx != -1:
            return fusion_idx

    return fusion_idx


def transform_simple_fusion_patterns(
    sorted_graph: List[Tensor], fusion_patterns: List[Any]
) -> List[Tensor]:
    output_tensors = []
    to_remove = set()
    has_modified = False
    for tensor in sorted_graph:
        if tensor in to_remove:
            to_remove.remove(tensor)
            continue

        if tensor._attrs["is_output"]:
            output_tensors.append(tensor)
            continue

        fusion_idx = _find_fusion_root(tensor, fusion_patterns)
        if fusion_idx == -1:
            continue

        to_remove_candidate = set()
        to_remove_dst_op = {}

        src_op = extract_only_one_op(tensor._attrs["src_ops"])
        inputs = list(src_op._attrs["inputs"])
        to_remove_dst_op[src_op] = list(inputs)
        src_op_num_inputs = len(inputs)

        last_tensor = tensor
        to_remove_candidate.add(last_tensor)

        for _ in range(len(fusion_patterns[fusion_idx][0]) - 1):
            # The check is done in _find_fusion_root, therefore we only need to
            # know how many steps to go forward.
            next_op = extract_only_one_op(last_tensor._attrs["dst_ops"])
            if next_op._attrs["op"] == "elementwise":
                next_op_inputs = next_op._attrs["args"]
            else:
                next_op_inputs = next_op._attrs["inputs"]
            assert (
                len(next_op_inputs) <= 2 and len(next_op_inputs) > 0
            ), "next_op in pattern should have input length of 1 or 2, got {} instead".format(
                len(next_op_inputs)
            )
            if len(next_op_inputs) == 2:
                # This is the case of add/mul/etc. we put them into inputs.
                if next_op_inputs[0] is last_tensor:
                    other_tensor = next_op_inputs[1]
                elif next_op_inputs[1] is last_tensor:
                    other_tensor = next_op_inputs[0]
                else:
                    raise AssertionError("input does not come from upstream node")
                inputs.append(other_tensor)

                if next_op in to_remove_dst_op:
                    to_remove_dst_op[next_op].append(other_tensor)
                else:
                    to_remove_dst_op[next_op] = [other_tensor]

            last_tensor = next_op._attrs["outputs"][0]
            to_remove_candidate.add(last_tensor)

        # A final check to make sure our replacement is valid.
        new_op = fusion_patterns[fusion_idx][1]

        # For bias_add fusion, use is_valid_inputs
        check_inputs_func = getattr(new_op, "is_valid_inputs", None)
        if check_inputs_func is not None:
            valid, _ = check_inputs_func(*inputs)
            if not valid:
                continue
        else:
            # gemm/conv epilogue fusion with elementwise ops doesn't
            # support broadcasting except for bias_add.
            # Here we do assume that all other inputs are elementwise inputs.
            cannot_fuse = False
            for elementwise_input in inputs[src_op_num_inputs:]:
                if tensor.shape() != elementwise_input.shape():
                    cannot_fuse = True
                    break
            if cannot_fuse:
                continue

        # inputs here might not be ready in graph. But we will toposort again
        # at end of pass so it's okay.
        has_modified = True
        new_tensor = new_op(**src_op._get_op_attributes())(*inputs)
        copy_tensor_attributes(new_tensor, last_tensor)
        if new_tensor._attrs["is_output"]:
            output_tensors.append(new_tensor)
        replace_tensor(last_tensor, new_tensor)
        for dst_op, tensors in to_remove_dst_op.items():
            remove_dst_op_from_tensor(tensors, dst_op)
        to_remove |= to_remove_candidate

    if has_modified:
        sorted_graph = toposort(output_tensors)
        sorted_graph = sanitize_sorted_graph(sorted_graph)
    return sorted_graph
