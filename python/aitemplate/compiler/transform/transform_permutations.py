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
from typing import List

import numpy as np

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.compiler.transform import transform_utils


NAME_TO_DIM = {
    "permute021": [0, 2, 1],
    "permute210": [2, 1, 0],
    "permute102": [1, 0, 2],
    "permute0213": [0, 2, 1, 3],
}


def get_permutation(op: Operator):
    if op._attrs["op"] == "permute":
        permutation = list(op._attrs["dims"])
    elif op._attrs["op"] in NAME_TO_DIM:
        permutation = NAME_TO_DIM[op._attrs["op"]]
    else:
        raise NotImplementedError(
            f"Not implemented for permute operation: {op._attrs['op']}"
        )
    return permutation


def remove_second_permutation_from_graph(
    permutation_1: Operator, permutation_2: Operator
):
    input_tensor_p1 = permutation_1._attrs["inputs"][0]
    input_tensor_p2 = permutation_2._attrs["inputs"][0]
    output_tensor = permutation_2._attrs["outputs"][0]

    input_tensor_p1._attrs["dst_ops"].update(output_tensor._attrs["dst_ops"])
    input_tensor_p2._attrs["dst_ops"].discard(permutation_2)

    for dst_op in output_tensor._attrs["dst_ops"]:
        dst_op.replace_input_tensor(output_tensor, input_tensor_p1)

    if output_tensor._attrs["is_output"]:
        input_tensor_p1._attrs["is_output"] = True
        input_tensor_p1._attrs["name"] = output_tensor._attrs["name"]

    transform_utils.remove_tensor_from_sorted_graph(output_tensor)


def _reshaped_or_strided_input_or_output_accessor(op: Operator) -> bool:
    def _reshaped_or_strided_tensor_accessor(accessor: TensorAccessor) -> bool:
        if (
            accessor.actual_shapes is not None
            and accessor.actual_shapes != accessor.original_shapes
        ):
            return True

        # Is it a strided accessor
        if hasattr(accessor, "stride_dim") and accessor.stride_dim is not None:
            return True

        return False

    input_accessors = op._attrs.get("input_accessors", None)
    output_accessors = op._attrs.get("output_accessors", None)

    return (
        (input_accessors is not None)
        and _reshaped_or_strided_tensor_accessor(input_accessors[0])
    ) or (
        (output_accessors is not None)
        and _reshaped_or_strided_tensor_accessor(output_accessors[0])
    )


def eliminate_permutations(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    if len(sorted_graph) < 2:
        return sorted_graph
    removed_op = set()
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        for cur_op in src_ops:
            if cur_op in removed_op:
                continue
            if not cur_op._attrs["op"].startswith("permute"):
                continue
            if _reshaped_or_strided_input_or_output_accessor(cur_op):
                continue
            curr_op_output = cur_op._attrs["outputs"][0]
            dst_ops = curr_op_output._attrs["dst_ops"]
            n_dst_ops = len(dst_ops)
            if n_dst_ops == 0:
                continue
            remove_list = []
            for next_op in dst_ops:
                if not next_op._attrs["op"].startswith("permute"):
                    continue
                if _reshaped_or_strided_input_or_output_accessor(next_op):
                    continue
                p1 = get_permutation(cur_op)
                p2 = get_permutation(next_op)
                if len(p1) != len(p2):
                    continue
                if not np.all(np.array(p1)[p2] == np.arange(0, len(p1))):
                    continue
                is_input = cur_op._attrs["inputs"][0]._attrs["is_input"]
                is_output = next_op._attrs["outputs"][0]._attrs["is_output"]
                if is_input and is_output:
                    continue
                remove_list.append(next_op)

            for next_op in remove_list:
                remove_second_permutation_from_graph(cur_op, next_op)
                removed_op.add(next_op)

            if len(remove_list) == n_dst_ops:
                transform_utils.remove_single_tensor_op_from_sorted_graph(cur_op)
                removed_op.add(cur_op)

    return transform_utils.sanitize_sorted_graph(sorted_graph)
