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
from typing import List, Optional

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor

from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.compiler.transform import transform_utils

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


def _make_first_cat_original_input_shapes(
    first_cat: Operator, target_tensor: Tensor, target_original_shape: IntVar
) -> List[IntVar]:
    """
    We build original shape for each input of the first op if the first op
    is consumed by a view op before being passed to the cat. For example,
    let's say the original graph looks like below:
        x0 = tensor([batch, 2 * 4])
        x1 = tensor([batch, 3 * 4])
        x2 = tensor([batch, 4, 4])
        concat_0 = concatenate([x0, x1], dim=1)
        reshape_1 = reshape(concat_0, [batch, 5, 4])
        concat_1 = concatenate([reshape_1, x2], dim=1)
    The transform_strided_op_and_view_op.py pass will fuse reshape_1 into
    concat_1. When we merge concat_0 and concat_1, we have to "re-build" the
    reshape (conceptually) for x0 and x1. Otherwise, we would end up with
    rank mismatch since rank(x0) and rank(x1) is 2 but rank(x2) is 3.
    This function tries to build the original input shape for x0 and x1 to
    make them look like they were reshaped before being consumed by concat_1.
    In this example, the original input shapes for x0 and x1 would be
    [batch, 2, 4] and [batch, 3, 4].
    """
    assert (
        first_cat._attrs["op"] == "concatenate"
    ), f"expected {first_cat=} to be a concatenate op"
    assert (
        len(first_cat._attrs["outputs"]) == 1
    ), f"expected a single output for {first_cat=}"
    if "original_inputs" in first_cat._attrs:
        original_inputs = first_cat._attrs["original_inputs"]
    else:
        original_inputs = first_cat._attrs["inputs"]
    original_input_shapes = [inp.shape() for inp in original_inputs]
    first_op_output = first_cat._attrs["outputs"][0]
    first_op_output_shape = first_op_output.shape()
    target_tensor_shape = target_tensor.shape()
    assert (
        first_op_output_shape == target_tensor_shape
    ), f"expected {first_op_output_shape=} and {target_tensor_shape=} to be the same"
    # We didn't fuse any view op into the cat op
    if target_tensor_shape == target_original_shape:
        return original_input_shapes
    # target_original_shape and original_inputs' shapes should meet all the
    # shape requirement since they were validated by transform_strided_op_and_view_op.
    # We just make some conservative checks here, which are not exact the same
    # as those performed by transform_strided_op_and_view_op. Instead, these
    # checks ensure the reshape operations are well-formed with respect to the cat ops.
    target_original_dynamic_dims = []
    target_original_static_numel = 1
    cat_dim = first_cat._attrs["concat_dim"]
    assert shape_utils.is_static_dimension(
        target_original_shape, cat_dim
    ), f"{cat_dim=} cannot be dynamic in {target_original_shape=}"
    for idx, orig_dim in enumerate(target_original_shape):
        if isinstance(orig_dim, IntImm):
            target_original_static_numel *= orig_dim._attrs["values"][0]
        else:
            target_original_dynamic_dims.append((idx, orig_dim))
    # it cannot happen if all dims are dynamic
    assert len(target_original_dynamic_dims) < len(
        target_original_shape
    ), f"not allowed to have all dynamic dims in {target_original_shape=}"

    first_op_output_name = first_op_output._attrs["name"]
    assert (
        first_op_output_name is not None
    ), f"expected the name of {first_op_output=} to be not None"
    original_cat_dim_val = target_original_shape[cat_dim]._attrs["values"][0]
    target_original_static_numel = target_original_static_numel // original_cat_dim_val
    for input_idx, orig_input_shape in enumerate(original_input_shapes):
        static_numel = 1
        dynamic_dims = []
        for dim_idx, dim in enumerate(orig_input_shape):
            if isinstance(dim, IntImm):
                static_numel *= dim._attrs["values"][0]
            else:
                dynamic_dims.append((dim_idx, dim))
        # make sure each input shape has the same dynamic dims as the original target shape
        assert (
            dynamic_dims == target_original_dynamic_dims
        ), f"{dynamic_dims=} is not the same as {target_original_dynamic_dims=}"
        new_input_shape = copy.deepcopy(target_original_shape)
        assert (
            static_numel >= target_original_static_numel
            and static_numel % target_original_static_numel == 0
        ), (
            f"invalid {target_original_static_numel=} and {static_numel=} for "
            f"{target_original_shape=} and {orig_input_shape=} at {input_idx=}"
        )
        dim_name = f"_ait_internal_{first_op_output_name}_dim_{input_idx}"
        new_input_shape[cat_dim] = IntImm(
            static_numel // target_original_static_numel, name=dim_name
        )
        original_input_shapes[input_idx] = new_input_shape
    return original_input_shapes


def _update_first_cat_input_accessor(
    first_cat_idx: int,
    first_cat_old_orig_shape: List[IntVar],
    first_cat_new_orig_shape: List[IntVar],
    new_orig_input: Tensor,
    new_first_cat_input_accessors: List[TensorAccessor],
) -> None:
    """
    A helper function that updates the input accessor with respect to the new shape
    """
    first_cat_input_accessor = new_first_cat_input_accessors[first_cat_idx]
    assert first_cat_old_orig_shape == first_cat_input_accessor.original_shapes, (
        f"expected {first_cat_old_orig_shape} to "
        f"be the same as {first_cat_input_accessor.original_shapes=}"
    )
    # reconstruct the input accessor with the new shape
    if first_cat_new_orig_shape != first_cat_input_accessor.original_shapes:
        assert (
            first_cat_input_accessor.stride_dim is None
        ), f"expected {first_cat_input_accessor.stride_dim=} to be None"
        new_tensor_accessor = TensorAccessor(new_orig_input)
        # we must inherit the actual_shapes from the old input_accessor
        # if it's set, because it comes from the root of the chained
        # concat ops
        new_actual_shapes = (
            first_cat_input_accessor.actual_shapes
            if first_cat_input_accessor.actual_shapes is not None
            else first_cat_input_accessor.original_shapes
        )
        new_tensor_accessor.update_base_shape(new_actual_shapes)
        new_first_cat_input_accessors[first_cat_idx] = new_tensor_accessor


def _try_merge_split_cat(split_op: Operator, cat: Operator) -> bool:
    # If split_op carries strided input_accessors, we skip it
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


def _try_merge_cat_cat(first_cat: Operator, cat: Operator) -> bool:
    # Make sure input_accessors do not carry any strided information.
    # It may happen. For example, an input of the cat can be of a strided
    # tensor generated by slice, which takes another concat's output.
    # Something like below:
    #     y1 = concat(x0, x1)
    #     y2 = slice(y1)
    #     y = cat(y1, y2)
    # In such a case, we cannot merge those two concat ops.
    if not all(
        accessor.stride_dim is None for accessor in cat._attrs["input_accessors"]
    ):
        return False
    # skip if the first cat has any fused strided op
    if any(mask is False for mask in first_cat._attrs["input_masks"]):
        return False
    # skip if the second cat has any fused strided op
    if any(mask is False for mask in cat._attrs["input_masks"]):
        return False
    # If first_cat carries strided input_accessors, we skip it
    if "input_accessors" in first_cat._attrs:
        if any(
            input_accessor.stride_dim is not None
            for input_accessor in first_cat._attrs["input_accessors"]
        ):
            return False
    first_cat_inputs = first_cat._attrs["inputs"]
    first_cat_outputs = first_cat._attrs["outputs"]
    assert (
        len(first_cat_outputs) == 1
    ), f"expected {first_cat_outputs=} to have a single output"
    first_cat_output = first_cat_outputs[0]
    cat_inputs = cat._attrs["inputs"]
    cat_original_inputs = cat._attrs["original_inputs"]
    new_cat_inputs = []
    new_cat_original_inputs = []
    new_cat_input_accessors = []
    i = 0
    for i, cat_input in enumerate(cat_inputs):
        if cat_input is first_cat_output:
            # If any view op has been fused to the first cat,
            # the original input shapes would be the reshape-to shapes if we
            # apply the same reshape (or other view ops) to the first cat op's
            # inputs. Otherwise, the original input shapes are just the original
            # shapes of those inputs
            first_cat_new_original_input_shapes = _make_first_cat_original_input_shapes(
                first_cat,
                cat._attrs["inputs"][i],
                cat._attrs["input_accessors"][i].original_shapes,
            )
            new_cat_inputs.extend(first_cat._attrs["inputs"])
            original_inputs = first_cat._attrs["original_inputs"]
            new_first_cat_input_accessors = copy.deepcopy(
                first_cat._attrs["input_accessors"]
            )
            dtype = first_cat._attrs["outputs"][0].dtype()
            for first_cat_idx, first_cat_new_orig_shape in enumerate(
                first_cat_new_original_input_shapes
            ):
                if first_cat_new_orig_shape == original_inputs[first_cat_idx].shape():
                    new_orig_input = original_inputs[first_cat_idx]
                else:
                    # This is the case when the first cat's output is applied to a
                    # view op. Note that we don't need to run toposort against this
                    # newly-created Tensor, because it's not part of the main graph.
                    # Instead, it's only in the merged cat's "original_inputs" attribute.
                    new_orig_input = Tensor(shape=first_cat_new_orig_shape, dtype=dtype)
                new_cat_original_inputs.append(new_orig_input)
                first_cat_old_orig_shape = original_inputs[first_cat_idx].shape()
                _update_first_cat_input_accessor(
                    first_cat_idx,
                    first_cat_old_orig_shape,
                    first_cat_new_orig_shape,
                    new_orig_input,
                    new_first_cat_input_accessors,
                )
            new_cat_input_accessors.extend(new_first_cat_input_accessors)
        else:
            new_cat_inputs.append(cat_input)
            new_cat_original_inputs.append(cat_original_inputs[i])
            new_cat_input_accessors.append(cat._attrs["input_accessors"][i])

    for tensor in new_cat_inputs:
        if tensor is first_cat_output:
            return False

    cat._attrs["inputs"] = new_cat_inputs
    # make sure all of the input_masks values are True. We may need to
    # change this part later when we have TensorAccessors, depending on
    # the order of the transformations.
    assert all(cat._attrs["input_masks"])
    cat._attrs["input_accessors"] = new_cat_input_accessors
    cat._attrs["original_inputs"] = list(new_cat_original_inputs)
    cat._attrs["input_masks"] = [True] * len(new_cat_inputs)
    for tensor in first_cat_inputs:
        tensor._attrs["dst_ops"].discard(first_cat)
        tensor._attrs["dst_ops"].add(cat)
    for tensor in first_cat_outputs:
        transform_utils.remove_tensor_from_sorted_graph(tensor)
    return True


FIRST_OP_CANDIDATES = {"split", "concatenate"}


def _get_first_op_candidate(tensor: Tensor) -> Optional[Operator]:
    """
    Return a candidate op that can be used as the first op.
    This method will return None if it cannot fine such a candidate.
    """
    src_ops = tensor._attrs["src_ops"]
    if len(src_ops) == 0:
        return None
    if len(src_ops) == 1:
        src_op = list(src_ops)[0]
        if src_op._attrs["op"] in FIRST_OP_CANDIDATES:
            return src_op
        else:
            return None
    cand_ops = [op for op in src_ops if op._attrs["op"] in FIRST_OP_CANDIDATES]
    # Let's handle a simple case first, where src_ops has a single concatenate op.
    # We could extend it to support more complicated cases.
    if len(cand_ops) != 1:
        return None
    src_op = list(cand_ops)[0]
    if src_op._attrs["op"] == "concatenate":
        return src_op
    else:
        return None


def _merge_split_and_cat(sorted_graph: List[Tensor]) -> List[Tensor]:  # noqa: C901
    to_be_merged_ops = []
    visited = set()
    for tensor in sorted_graph:
        cand_op = _get_first_op_candidate(tensor)
        if cand_op is None or cand_op in visited:
            continue
        first_op = cand_op

        cat = None
        found_cat_op = True
        for output_t in first_op._attrs["outputs"]:
            if len(output_t._attrs["dst_ops"]) > 1:
                found_cat_op = False
                break
            # If first op is output, it can't be fused.
            if output_t._attrs["is_output"]:
                found_cat_op = False
                continue
            next_ops = output_t._attrs["dst_ops"]
            if len(next_ops) != 1:
                break
            next_op = list(next_ops)[0]
            if next_op._attrs["op"] != "concatenate":
                found_cat_op = False
                break
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
        visited.add(first_op)
        visited.add(cat)

    for ops in to_be_merged_ops:
        first_op_type = ops[0]._attrs["op"]
        if first_op_type == "split":
            _try_merge_split_cat(ops[0], ops[1])
        elif first_op_type == "concatenate":
            _try_merge_cat_cat(ops[0], ops[1])
        else:
            assert False, f"unsupported {first_op_type=} for merging with cat"

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
