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
Perform memory operator related transformations.
"""
import copy
from typing import List

from aitemplate.compiler.base import Operator, Tensor

from aitemplate.compiler.ops.tensor.dynamic_slice import dynamic_slice
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.compiler.transform import transform_strided_ops_utils, transform_utils
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.compiler.transform.transform_merge_slice_ops import merge_slice_ops

from aitemplate.utils import graph_utils, shape_utils


def _eliminate_cat(sorted_graph: List[Tensor]) -> List[Tensor]:
    # If we only have a single cat op in the graph, let's keep it.
    # This almost always comes from unit tests.
    if len(graph_utils.get_sorted_ops(sorted_graph)) <= 1:
        return sorted_graph

    single_input_cat_ops = []
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in sorted_ops:
        if op._attrs["op"] != "concatenate":
            continue
        if len(op._attrs["outputs"]) != 1:
            continue
        if len(op._attrs["inputs"]) == 0:
            op._attrs["outputs"][0]._attrs["src_ops"].remove(op)
            op._attrs["outputs"] = []
            continue
        if (len(op._attrs["inputs"]) == 1) and (False not in op._attrs["input_masks"]):
            single_input_cat_ops.append(op)

    for op in single_input_cat_ops:
        input_tensor = op._attrs["inputs"][0]
        output_tensor = op._attrs["outputs"][0]
        # tensor can not be input and output
        if output_tensor._attrs["is_output"] and input_tensor._attrs["is_input"]:
            continue
        transform_utils.remove_single_tensor_op_from_sorted_graph(op)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _update_cat_dst_ops(
    first_cat: Operator, second_cat: Operator, cat_dim_offset: int
) -> None:
    """
    Add all the strided dst_ops of the first cat to the second and
    make an appropriate slice op between the second cat and each dst_ops.
    cat_dim_offset represents the offset of the first cat output appearing
    in the second cat along the cat_dim dimension.
    """
    first_cat_output = first_cat._attrs["outputs"][0]
    first_cat_dst_ops = first_cat_output._attrs["dst_ops"]
    # the first cat does not have any strided ops
    if len(first_cat_dst_ops) <= 1:
        return
    first_cat_shape = first_cat_output.shape()
    rank = len(first_cat_shape)
    cat_dim = first_cat._attrs["concat_dim"]
    assert transform_strided_ops_utils.cat_split_dim_is_static(
        first_cat, cat_dim
    ), f"expected the {cat_dim=} of {first_cat=} to be static"
    second_cat_output = second_cat._attrs["outputs"][0]
    # make start_indices and end_indices for the slice
    for idx, first_cat_dst_op in enumerate(first_cat_dst_ops):
        if first_cat_dst_op is second_cat:
            continue
        else:
            # Make a new slice op. Note that it's fine we make a new slice op from
            # another slice op, because consecutive slice ops will be merged
            # by the merge_slice_ops pass
            slice_start_indices = [0] * rank
            slice_end_indices = [None] * rank
            slice_start_indices[cat_dim] = cat_dim_offset
            slice_end_indices[cat_dim] = (
                cat_dim_offset + first_cat_shape[cat_dim].value()
            )
            slice_op = dynamic_slice()
            slice_op_name = f'dynamic_slice_{idx}_{first_cat._attrs["name"]}'
            slice_op._attrs["name"] = slice_op_name
            slice_op._attrs["original_name"] = slice_op_name
            slice_output = slice_op(
                second_cat_output, slice_start_indices, slice_end_indices
            )
            slice_output._attrs["name"] = f"{slice_op_name}_0"
            slice_output._attrs["dst_ops"].add(first_cat_dst_op)
            # remove the old strided op from first cat's dst_ops
            first_cat_dst_ops.remove(first_cat_dst_op)
            # update the strided op's input to the newly-created slice output
            first_cat_dst_op.replace_input_tensor(first_cat_output, slice_output)


def _is_supported_dst_op_for_first_cat(
    dst_op: Operator,
) -> bool:
    """
    A helper function that returns True if the given dst_op is
    * a supported strided op; or
    * a view op that is only used by a supported stride op; or
    * a view op that is indirectly (via another single-dst view op) used
      by a supported strided op.
    Note that technically, this checking is not necessary, because we could
    let other passes process the likely fusion patterns related to
    concat + strided_op. However, it seems to be safer if we could add
    more tests similar to test_fuse_strided_cat_reshape_cat but with different
    strided ops such as gemm/layernorm/etc. To be conservative, we only
    enable the following patterns and will remove the restriction once we
    have more test coverage.
    """
    view_ops = ["reshape", "flatten", "dynamic_slice", "squeeze", "unsqueeze"]
    # FIXME: enable other ops with input_accessors
    supported_strided_ops = ["elementwise", "fused_elementwise"]

    def _supported_op_type(op_type):
        if op_type in supported_strided_ops:
            return True
        return op_type.startswith("bmm_crr")

    dst_op_type = dst_op._attrs["op"]
    if _supported_op_type(dst_op_type):
        return True
    while dst_op_type in view_ops:
        dst_op_outputs = dst_op._attrs["outputs"]
        if len(dst_op_outputs) != 1:
            return False
        dst_op_output = dst_op_outputs[0]
        if dst_op_output._attrs["is_output"]:
            return False
        next_dst_ops = dst_op_output._attrs["dst_ops"]
        if len(next_dst_ops) != 1:
            return False
        dst_op = next_dst_ops[0]
        dst_op_type = dst_op._attrs["op"]
        if _supported_op_type(dst_op_type):
            return True
    return False


def _check_first_cat(first_cat: Operator, second_cat: Operator) -> bool:
    """
    return True if the first cat is valid for fusion
    """
    # Make sure input_accessors do not carry any strided information.
    # It may happen. For example, an input of the cat can be of a strided
    # tensor generated by slice, which takes another concat's output.
    # Something like below:
    #     y1 = concat(x0, x1)
    #     y2 = slice(y1)
    #     y = cat(y1, y2)
    # In such a case, we cannot merge those two concat ops.
    if not all(
        accessor.actual_shapes is None
        for accessor in first_cat._attrs["input_accessors"]
    ):
        return False
    if not all(first_cat._attrs["input_masks"]):
        return False

    # we need to make sure all other dst ops except the second cat have input
    # accessors for which we may generate valid strided information. We will
    # leverage the input accessor by injecting a slice op between the merged
    # cat and the strided op (e.g. add).
    cat_dim = first_cat._attrs["concat_dim"]
    first_cat_outputs = first_cat._attrs["outputs"]
    assert (
        len(first_cat_outputs) == 1
    ), f"expected {first_cat_outputs=} to have a single output"
    first_cat_output = first_cat_outputs[0]
    first_cat_dst_ops = first_cat_output._attrs["dst_ops"]
    if len(first_cat_dst_ops) == 1:
        return True
    if not transform_strided_ops_utils.cat_split_dim_is_static(first_cat, cat_dim):
        return False
    # we cannot leverage slice if any of the dimensions after cat_dim is dynamic
    if not shape_utils.all_static_dimensions(first_cat_output.shape(), cat_dim):
        return False

    # we can fuse the first cat into the second only if all of the first cat's
    # dst ops are valid
    for dst_op in first_cat_dst_ops:
        if dst_op is second_cat:
            continue
        if not _is_supported_dst_op_for_first_cat(dst_op):
            return False
        # merging first_cat and second_cat may introduce a cycle
        if transform_utils.is_ancestor(dst_op, second_cat):
            return False
    return True


def _check_second_cat(cat: Operator) -> bool:
    """
    return True if the second cat is valid for fusion
    """
    if len(cat._attrs["outputs"]) != 1:
        return False
    # Similar to the first cat, make sure the second cat's input_accessors
    # do not carry any strided information.
    if not all(
        accessor.actual_shapes is None for accessor in cat._attrs["input_accessors"]
    ):
        return False
    if not all(cat._attrs["input_masks"]):
        return False
    return True


def _try_merge_cat_cat(first_cat: Operator, second_cat: Operator) -> bool:
    if not _check_first_cat(first_cat, second_cat):
        return False
    if not _check_second_cat(second_cat):
        return False
    first_cat_inputs = first_cat._attrs["inputs"]
    first_cat_outputs = first_cat._attrs["outputs"]
    first_cat_output = first_cat_outputs[0]
    second_cat_inputs = second_cat._attrs["inputs"]
    second_cat_original_inputs = second_cat._attrs["original_inputs"]
    new_cat_inputs = []
    new_cat_original_inputs = []
    new_cat_input_accessors = []
    for i, second_cat_input in enumerate(second_cat_inputs):
        if second_cat_input is first_cat_output:
            new_cat_inputs.extend(first_cat._attrs["inputs"])
            first_cat_original_inputs = first_cat._attrs["inputs"]
            new_cat_original_inputs.extend(first_cat_original_inputs)
            new_cat_input_accessors.extend(
                copy.deepcopy(first_cat._attrs["input_accessors"])
            )
        else:
            new_cat_inputs.append(second_cat_input)
            new_cat_original_inputs.append(second_cat_original_inputs[i])
            new_cat_input_accessors.append(second_cat._attrs["input_accessors"][i])

    for tensor in new_cat_inputs:
        if tensor in first_cat_outputs:
            return False

    # note that we have to compute cat_dim_offset before updating cat's inputs,
    # because we determine the cat_dim_offset based on its old inputs
    cat_dim_offset = 0
    cat_dim = second_cat._attrs["concat_dim"]
    for second_cat_input in second_cat._attrs["inputs"]:
        if second_cat_input is first_cat_output:
            break
        cat_dim_offset += second_cat_input._size(cat_dim).value()

    second_cat._attrs["inputs"] = new_cat_inputs
    # make sure all of the input_masks values are True. We may need to
    # change this part later when we have TensorAccessors, depending on
    # the order of the transformations.
    assert all(second_cat._attrs["input_masks"])
    second_cat._attrs["input_accessors"] = new_cat_input_accessors
    second_cat._attrs["original_inputs"] = list(new_cat_original_inputs)
    second_cat._attrs["input_masks"] = [True] * len(new_cat_inputs)
    for tensor in first_cat_inputs:
        # the same tensor may be used multiple times
        tensor._attrs["dst_ops"].discard(first_cat)
        tensor._attrs["dst_ops"].add(second_cat)
    # now we can move strided ops from the first cat to the merged cat with
    # an appropriate slice op between the merged cat and each strided op
    _update_cat_dst_ops(first_cat, second_cat, cat_dim_offset)
    transform_utils.remove_tensor_from_sorted_graph(first_cat_output)
    return True


def _try_merge_split_cat(split_op: Operator, cat: Operator) -> bool:
    # If split_op carries strided input_accessors, we skip it.
    if not all(
        accessor.actual_shapes is None for accessor in cat._attrs["input_accessors"]
    ):
        return False
    if not all(cat._attrs["input_masks"]):
        return False
    split_op_inputs = split_op._attrs["inputs"]
    split_op_outputs = split_op._attrs["outputs"]
    cat_inputs = cat._attrs["inputs"]
    cat_original_inputs = cat._attrs["original_inputs"]
    new_cat_inputs = []
    new_cat_original_inputs = []
    new_cat_input_accessors = []
    i = 0
    while i < len(cat_inputs):
        matched = True
        for j, _ in enumerate(split_op_outputs):
            if (i + j >= len(cat_inputs)) or (
                cat_inputs[i + j] is not split_op_outputs[j]
            ):
                matched = False
                break
        if matched:
            # split doens't have "original_inputs" attribute
            split_op_inputs = split_op._attrs["inputs"]
            new_cat_inputs.extend(split_op_inputs)
            new_cat_original_inputs.extend(split_op_inputs)
            new_cat_input_accessors.extend([TensorAccessor(t) for t in split_op_inputs])
            i += len(split_op_outputs)
        else:
            new_cat_inputs.append(cat_inputs[i])
            new_cat_original_inputs.append(cat_original_inputs[i])
            new_cat_input_accessors.append(cat._attrs["input_accessors"][i])
            i += 1

    for tensor in new_cat_inputs:
        if tensor in split_op_outputs:
            return False

    cat._attrs["inputs"] = new_cat_inputs
    # make sure all of the input_masks values are True. We may need to
    # change this part later when we have TensorAccessors, depending on
    # the order of the transformations.
    assert all(cat._attrs["input_masks"])
    cat._attrs["input_accessors"] = new_cat_input_accessors
    cat._attrs["original_inputs"] = list(new_cat_original_inputs)
    cat._attrs["input_masks"] = [True] * len(new_cat_inputs)
    for tensor in split_op_inputs:
        tensor._attrs["dst_ops"].discard(split_op)
        tensor._attrs["dst_ops"].add(cat)
    for tensor in split_op_outputs:
        transform_utils.remove_tensor_from_sorted_graph(tensor)
    return True


FIRST_OP_CANDIDATES = {"split", "concatenate"}


def _merge_split_and_cat(sorted_graph: List[Tensor]) -> List[Tensor]:  # noqa: C901
    to_be_merged_ops = []
    visited = set()
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] not in FIRST_OP_CANDIDATES:
            continue
        if src_op in visited:
            continue
        first_op = src_op

        cat = None
        found_cat_op = True
        for output_t in first_op._attrs["outputs"]:
            # TODO: currently, we only allow concatenate output with multiple dst_ops.
            # We may need to extend it to split ops.
            if (
                len(output_t._attrs["dst_ops"]) > 1
                and first_op._attrs["op"] != "concatenate"
            ):
                found_cat_op = False
                break
            # If first op is output, it can't be fused.
            if output_t._attrs["is_output"]:
                found_cat_op = False
                continue
            next_ops = output_t._attrs["dst_ops"]
            if len(next_ops) == 0:
                break
            next_concats = [n for n in next_ops if n._attrs["op"] == "concatenate"]
            # only support cases where first_cat is consumed by a single concat
            if len(next_concats) != 1:
                found_cat_op = False
                break
            next_op = next_concats[0]
            if cat is None:
                cat = next_op
            if next_op is not cat:
                found_cat_op = False
                break

        if cat is None or not found_cat_op:
            continue

        first_op_dim = (
            first_op._attrs["concat_dim"]
            if first_op._attrs["op"] == "concatenate"
            else first_op._attrs["split_dim"]
        )
        if cat._attrs["concat_dim"] != first_op_dim:
            continue

        to_be_merged_ops.append([first_op, cat])
        # only add first_op to the visited set to cases where
        # we may have chained concat cases:
        #     concat_0 = concat(x0...)
        #     concat_1 = concat(concat_0...)
        #     concat_2 = concat(concat_1...)
        # where merging concat_0 and concat_1 is invalid but merging concat_1
        # and concat_2 is valid. If we include both first_op and cat into
        # the visited set, we would miss the opportunity of merging concat_1
        # and concat_2.
        visited.add(first_op)

    updated_cat_cat = False
    for ops in to_be_merged_ops:
        first_op_type = ops[0]._attrs["op"]
        if first_op_type == "split":
            _try_merge_split_cat(ops[0], ops[1])
        elif first_op_type == "concatenate":
            if _try_merge_cat_cat(ops[0], ops[1]):
                updated_cat_cat = True
        else:
            raise AssertionError(f"unsupported {first_op_type=} for merging with cat")

    # we adjusted input/output dependencies so need to run toposort again
    if updated_cat_cat:
        sorted_graph = toposort(sorted_graph)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _eliminate_split_full_idx(sorted_graph: List[Tensor]) -> List[Tensor]:
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] != "split":
            continue
        split_op = src_op
        dim = split_op._attrs["split_dim"]
        split_sizes = split_op._attrs["split_sizes"]
        assert len(split_op._attrs["inputs"]) == 1
        shape = split_op._attrs["inputs"][0]._attrs["shape"]
        if (
            len(split_sizes) == 1
            and shape_utils.is_static_dimension(shape, dim)
            and shape[dim]._attrs["values"][0] == split_sizes[0]
        ):
            input_tensor = split_op._attrs["inputs"][0]
            output_tensor = split_op._attrs["outputs"][0]
            # tensor can not be input and output
            if output_tensor._attrs["is_output"] and input_tensor._attrs["is_input"]:
                continue
            transform_utils.remove_single_tensor_op_from_sorted_graph(split_op)

    sorted_graph = transform_utils.sanitize_sorted_graph(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def transform_memory_ops(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """
    Eliminates unnecessary cat / split ops.
    """

    funcs = [
        _eliminate_split_full_idx,
        _merge_split_and_cat,
        merge_slice_ops,
        _eliminate_cat,
    ]
    num_ops = None
    should_continue = True
    while should_continue:
        for func in funcs:
            sorted_graph = func(sorted_graph)
        new_num_ops = len(graph_utils.get_sorted_ops(sorted_graph))
        if num_ops == new_num_ops:
            should_continue = False
        num_ops = new_num_ops
    return sorted_graph
