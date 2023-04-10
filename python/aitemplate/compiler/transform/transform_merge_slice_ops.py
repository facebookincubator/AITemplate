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
This file implements a pass that merges consecutive slice ops if possible.
"""
from typing import List, Optional

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor

from aitemplate.compiler.ops.tensor.dynamic_slice import MAX_INT32
from aitemplate.compiler.transform import transform_utils

from aitemplate.utils import shape_utils


def _try_merge_slice_slice(
    first_slice: Operator, second_slice: Operator, slice_dim: int
) -> bool:
    """
    This function tries to merge two consecutive slice ops with the following
    steps:
        * update the start_indices and end_indices fields of the second_slice
        * remove the first slice
    """
    first_slice_output = first_slice._attrs["outputs"][0]
    first_slice_input_shape = first_slice._attrs["inputs"][0].shape()
    second_slice_output = second_slice._attrs["outputs"][0]
    second_slice_output_shape = second_slice_output.shape()
    # note that all the dims of input_shape[slice_dim:] and output_shape[slice_dim:]
    # are static at this point
    for idx in range(slice_dim, first_slice_output._rank()):
        first_slice_dim_offset = first_slice._attrs["start_indices"][idx]
        # update the start and end indices of the second slice op
        new_start = second_slice._attrs["start_indices"][idx] + first_slice_dim_offset
        first_slice_input_dim = first_slice_input_shape[idx].value()
        # new start index exceeds the corresponding dim value of the first slice input shape
        if new_start >= first_slice_input_dim:
            return False
        new_end = new_start + second_slice_output_shape[idx].value()
        # new end index exceeds the corresponding dim value of the first slice input shape
        if new_end > first_slice_input_dim:
            return False
        first_slice_end = first_slice._attrs["end_indices"][idx]
        second_slice_end = second_slice._attrs["end_indices"][idx]
        if first_slice_end == MAX_INT32 == second_slice_end:
            new_end = MAX_INT32
        second_slice._attrs["start_indices"][idx] = new_start
        second_slice._attrs["end_indices"][idx] = new_end
    # remove the old strided op from the first cat's dst_ops
    transform_utils.remove_single_tensor_op_from_sorted_graph(first_slice)
    return True


def _check_slice_op(slice_op: Operator, slice_dim: int) -> bool:
    """
    Return True if the slice_op's indices are valid for being merged
    """
    slice_shape = slice_op._attrs["outputs"][0].shape()
    if not shape_utils.all_static_dimensions(slice_shape, slice_dim):
        return False
    # we expect normalized start_indices and end_indices
    start_index = slice_op._attrs["start_indices"][slice_dim]
    if start_index is None or start_index < 0:
        return False
    end_index = slice_op._attrs["end_indices"][slice_dim]
    if end_index is None or end_index < 0 or end_index <= start_index:
        return False
    return True


def _get_rightmost_non_dynamic_dim(shape: List[IntVar]) -> Optional[int]:
    """
    Return the index of the rightmost non-dynamic dim. For example, given
    a shape [3, dyn_dim, 4, 1], it would return 2, which is the index of the
    third dim.
    Return None if shape[-1] is dynamic.
    """
    idx = 0
    for dim in reversed(shape):
        if not isinstance(dim, IntImm):
            break
        idx += 1
    if idx == 0:
        return None
    return len(shape) - idx


def merge_slice_ops(sorted_graph: List[Tensor]) -> List[Tensor]:
    # a list of tuple(first_slice, second_slice, slice_dim)
    to_be_merged = []
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] != "dynamic_slice":
            continue
        first_slice = src_op
        first_slice_output = first_slice._attrs["outputs"][0]
        if first_slice_output._attrs["is_output"]:
            continue
        slice_dim = _get_rightmost_non_dynamic_dim(first_slice_output.shape())
        if slice_dim is None:
            continue
        if not _check_slice_op(first_slice, slice_dim):
            continue
        next_ops = first_slice_output._attrs["dst_ops"]
        if len(next_ops) != 1:
            continue
        next_op = next_ops[0]
        if next_op._attrs["op"] != "dynamic_slice":
            continue
        second_slice = next_op
        second_slice_output = second_slice._attrs["outputs"][0]
        if first_slice_output._rank() != second_slice_output._rank():
            continue
        second_slice_dim = _get_rightmost_non_dynamic_dim(second_slice_output.shape())
        if slice_dim != second_slice_dim:
            continue
        if not _check_slice_op(second_slice, slice_dim):
            continue
        to_be_merged.append([first_slice, second_slice, slice_dim])

    for first_slice, second_slice, slice_dim in to_be_merged:
        _try_merge_slice_slice(first_slice, second_slice, slice_dim)
    return transform_utils.sanitize_sorted_graph(sorted_graph)
