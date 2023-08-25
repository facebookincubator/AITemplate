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
import itertools
import logging
from collections import defaultdict
from typing import Dict, List

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.transform import transform_utils
from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)


def _fused_elementwise_ops_are_equal(op1: Operator, op2: Operator) -> bool:
    """We consider two fused elementwise to be duplicates when:
    2. Their elementwise operations are the same.
    1. And their inputs accessors are the same.

    NOTE: We assume the inputs are in the same order as the sub-elementwise
    operations. Otherwise, this is a problem because some elementwise operations
    are non-commutative.
    """
    op1_elementwise_ops = op1._attrs["elementwise_ops"]
    op2_elementwise_ops = op2._attrs["elementwise_ops"]
    op1_inputs, op2_inputs = op1._attrs["inputs"], op2._attrs["inputs"]
    op1_input_accessors = op1._attrs["input_accessors"]
    op2_input_accessors = op2._attrs["input_accessors"]
    if (
        len(op1_elementwise_ops) != len(op2_elementwise_ops)
        or len(op1_inputs) != len(op2_inputs)
        or len(op1_input_accessors) != len(op2_input_accessors)
    ):
        return False

    are_elementwise_equal = all(
        a._attrs["func"] == b._attrs["func"]
        for a, b in zip(op1_elementwise_ops, op2_elementwise_ops)
    )
    are_input_accessors_equal = all(
        input1 == input2 and input_accessor1 == input_accessor2
        for input1, input2, input_accessor1, input_accessor2 in zip(
            op1_inputs, op2_inputs, op1_input_accessors, op2_input_accessors
        )
    )
    return are_elementwise_equal and are_input_accessors_equal


def find_duplicate_fused_elementwise(
    sorted_graph: List[Tensor],
) -> Dict[Operator, List[Operator]]:
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    fused_elementwise_ops = filter(
        lambda operator: operator._attrs["op"] == "fused_elementwise", sorted_ops
    )
    visited = set()
    fusion_groups = defaultdict(list)

    for op1, op2 in itertools.combinations(fused_elementwise_ops, 2):
        if op1 in visited or op2 in visited:
            continue
        if _fused_elementwise_ops_are_equal(op1, op2):
            fusion_groups[op1].append(op2)
            visited.add(op2)

    return fusion_groups


def fuse_duplicate_fused_elementwise(
    sorted_graph: List[Tensor], _workdir: str
) -> List[Tensor]:
    """This pass finds all duplicate fused elementwise ops and fuses them once
    more. It assumes any fuse elementwise passes are complete.

    We do the fusion by taking all the duplicate fused elementwise ops and
    effectively deleting all but one. We make sure to transfer the outputs and
    output_accessors of the duplicate ops to the remaining op. That means, the
    newly fused op will have multiple outputs.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph

    _workdir : str
        Required by optimize_graph.py

    Returns
    ----------
    sorted_graph : List[Tensor]
        Modified input graph with duplicate fused elementwise ops fused together.
    """

    fusion_groups = find_duplicate_fused_elementwise(sorted_graph)
    for primary_op, duplicate_ops in fusion_groups.items():
        # Primary op inherits the outputs from the duplicate ops.

        for key in ("outputs", "output_accessors"):
            duplicate_ops_outputs = [
                output for op in duplicate_ops for output in op._attrs[key]
            ]
            primary_op._attrs[key] += duplicate_ops_outputs
            if key != "outputs":
                continue

            # Make sure to update src_ops in the output tensors.
            for output_tensor in duplicate_ops_outputs:
                old_src_ops = output_tensor._attrs["src_ops"]
                output_tensor._attrs["src_ops"] = set(old_src_ops) - set(
                    duplicate_ops
                ) | {primary_op}

            # Make sure to update dst_ops in the input tensors.
            for input_tensor in primary_op._attrs["inputs"]:
                input_tensor._attrs["dst_ops"] = set(
                    input_tensor._attrs["dst_ops"]
                ) - set(duplicate_ops)

        # Assumption: If the input accessors are the same, then the output's
        # original shape must be the same.
        prev_shape = primary_op._attrs["output_accessors"][0].original_shapes
        for output_accessor in primary_op._attrs["output_accessors"]:
            shape = output_accessor.original_shapes
            assert (
                prev_shape == shape
            ), "Output shapes mismatch in fuse_duplicate_fused_elementwise: {}, {}".format(
                prev_shape, shape
            )
            prev_shape = shape

        _LOGGER.info(
            "Fusing {} with {}".format(
                primary_op._attrs["name"],
                ", ".join([dup_op._attrs["name"] for dup_op in duplicate_ops]),
            )
        )

    return transform_utils.sanitize_sorted_graph(sorted_graph)
