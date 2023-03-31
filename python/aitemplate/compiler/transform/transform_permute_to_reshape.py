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
Transform permute to reshape wherever applicable.
"""
from typing import List

from aitemplate.compiler.base import IntImm, Operator, Tensor
from aitemplate.compiler.ops import reshape
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.toposort import toposort

from aitemplate.utils import graph_utils


def _check_permute_to_reshape(op: Operator) -> bool:
    """Check if applicable to replace permute with reshape.

    Args:
        op (Operator): reshape op

    Returns:
        bool: False if operation is not a permute or a permute with memory
            layout modification otherwise True.
    """
    if not op._attrs["op"].startswith("permute"):
        return False

    inputs = op._attrs["inputs"]

    assert (
        len(inputs) == 1
    ), "Permute operation {} should have 1 input, got {} instead".format(
        op._attrs["op"], len(inputs)
    )

    input_shape = inputs[0].shape()

    if op._attrs["op"] == "permute":
        permutation = list(op._attrs["dims"])
    elif op._attrs["op"] == "permute021":
        n_dims = len(input_shape)
        permutation = list(range(n_dims - 2)) + [n_dims - 1, n_dims - 2]
    elif op._attrs["op"] == "permute102":
        permutation = [1, 0, 2]
    elif op._attrs["op"] == "permute210":
        permutation = [2, 1, 0]
    elif op._attrs["op"] == "permute0213":
        permutation = [0, 2, 1, 3]
    else:
        raise NotImplementedError(
            f"Not implemented for permute operation: {op._attrs['op']}"
        )

    # Get non-singular dimension indices
    permutation = [
        dim_idx
        for dim_idx in permutation
        if not isinstance(input_shape[dim_idx], IntImm)
        or input_shape[dim_idx].value() != 1
    ]
    is_reshape = permutation == sorted(permutation)
    return is_reshape


def transform_permute_to_reshape(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Convert permute to reshape wherever applicable.

    When permute op involves moving one or more dimensions with size
    1 around where the order of non-singular dimensions is preserved,
    it's basically a reshape op, i.e. the underlying memory layout
    does not change.

    Example:
        [256x5x1x32] -> [256x5x32x1] (with 0132) is a reshape
        [256x1x5x1x32] -> [256x5x32x1x1] (with 02431) is a reshape
        [256x5x1x32] -> [256x32x5x1] (with 0312) is not a reshape

    Args:
        sorted_graph (List[Tensor]): input graph
        workdir (str, optional): current workdir for dumping debug info. Defaults to None.

    Returns:
        List[Tensor]: optimized graph
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)

    has_modified = False
    for op in sorted_ops:
        if not _check_permute_to_reshape(op):
            continue

        has_modified = True

        permute_input = op._attrs["inputs"][0]
        permute_output = op._attrs["outputs"][0]
        output_shape = permute_output.shape()

        transform_utils.remove_dst_op_from_tensor(permute_input, op)

        reshape_op = reshape()
        reshape_output = reshape_op(permute_input, output_shape)

        transform_utils.replace_tensor(permute_output, reshape_output)

        sorted_graph.append(reshape_output)

    if has_modified:
        sorted_graph = toposort(sorted_graph)
        transform_utils.sanitize_sorted_graph(sorted_graph)
    return sorted_graph
