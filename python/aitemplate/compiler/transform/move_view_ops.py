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
This pass move any view op between two concatenate ops to the front of the
first concatenate op if possible.
"""
import copy
from typing import Callable, List, Optional, Tuple

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.toposort import toposort

from aitemplate.utils import shape_utils


# TODO: support other view ops such as squeeze and unsqueeze
_SUPPORTED_VIEW_OPS = ["reshape", "flatten"]


def _make_input_view_shape(
    cat_input: Tensor,
    original_view_shape: List[IntVar],
    cat_dim: int,
    input_idx: int,
) -> Optional[List[IntVar]]:
    """
    Assumes that there is a pattern like concat + view_op in the graph, we tries
    to transform it into view_op + concat. However, it's not always valid to
    perform such a transformation, because the concat's original inputs may
    not be shape-compatible with the moved view_op. Currently, we only support
    cases where a view_op only changes the dims after cat_dim and the dims after
    the cat_dim must be static.

    For example, for the following code:

        x1 = Tensor(batch, 3 * 4)
        x2 = Tensor(batch, 5 * 4)
        concat_0 = concat([x1, x2], cat_dim=1)
        reshape_1 = reshape(concat_0, [batch, 8, 4])

    This function will generate shape [batch, 3, 4] for x1 and [batch, 5, 4]
    for x2, respectively.

    In contrast, if we have code like below:

        x1 = tensor([batch, 16])
        x2 = tensor([batch, 8])
        cat_1 = concatenate([x1, x2], cat_dim=1)
        reshape_2 = reshape(cat_1, [batch, 4, 6])

    We would return None for both x1 and x2, because we cannot make valid reshape
    ops for x1 and x2 while keeping the original semantics.

    Parameters
    ----------
    cat_input: Tensor
        a concat op's input for which we will generate a view op, e.g. the x1 or
        x2 tensor in the example above
    original_view_shape: List[IntVar]
        the shape of the view op's output, where the view op consumes the concat's
        output, e.g. the reshape op in the example above
    cat_dim: int
        the value of the cat_dim attribute of the concate op
    input_idx: int
        the index of the cat_input in the concat's inputs list
    """
    cat_input_shape = cat_input.shape()
    if cat_dim >= len(cat_input_shape) or cat_dim >= len(original_view_shape):
        return None
    # make sure each dimension at the same index in front of the cat_dim is the same for
    # both cat_input_shape and original_view_shape
    for curr_cat_dim, orig_dim in zip(
        cat_input_shape[:cat_dim], original_view_shape[:cat_dim]
    ):
        if curr_cat_dim != orig_dim:
            return None
    input_stride_at_cat_dim = shape_utils.get_static_stride(cat_input_shape, cat_dim)
    # make sure all dimensions are static after cat_dim
    if input_stride_at_cat_dim is None:
        return None
    orig_view_stride_at_cat_dim = shape_utils.get_static_stride(
        original_view_shape, cat_dim
    )
    # make sure all dimensions are static after cat_dim
    if orig_view_stride_at_cat_dim is None:
        return None
    new_input_view_shape = copy.deepcopy(original_view_shape)
    cat_stride = cat_input_shape[cat_dim].value() * input_stride_at_cat_dim
    if cat_stride % orig_view_stride_at_cat_dim != 0:
        return None
    orig_dim_name = original_view_shape[cat_dim]._attrs["name"]
    new_input_view_shape[cat_dim] = IntImm(
        cat_stride // orig_view_stride_at_cat_dim,
        name=f'{orig_dim_name}_{cat_input._attrs["name"]}_{input_idx}',
    )
    return new_input_view_shape


def _call_view_op(
    view_op: Callable, view_output_shape: List[IntVar], input_tensor: Tensor
) -> Tensor:
    """
    call the view_op with suitable arguments and return the output tensor
    """
    view_op_type = view_op._attrs["op"]
    if view_op_type == "reshape":
        output = view_op(input_tensor, view_output_shape)
    elif view_op_type == "flatten":
        output = view_op(input_tensor)
    else:
        raise AssertionError(f"unsupported {view_op_type=}")
    return output


def _try_move_view_op(
    first_cat: Operator,
    second_cat: Operator,
    view_op: Operator,
) -> bool:
    """
    Try to move the view_op to the front of the first_cat.
    Return true if the transformation is successful, False otherwise.
    """
    cat_dim = first_cat._attrs["concat_dim"]
    first_cat_output = first_cat._attrs["outputs"][0]
    first_cat_output_shape = first_cat_output.shape()
    # we might be able to support dynamic cat_dim, but let's be conservative
    # for now
    if not shape_utils.is_static_dimension(first_cat_output_shape, cat_dim):
        return False
    if second_cat._attrs["concat_dim"] != cat_dim:
        return False
    second_cat_output = second_cat._attrs["outputs"][0]
    if not shape_utils.is_static_dimension(second_cat_output.shape(), cat_dim):
        return False
    # We are not always able to move the view op. For example, we cannot
    # move the reshape to the front of cat_1 in the following code:
    #    x1 = tensor([batch, 16])
    #    x2 = tensor([batch, 8])
    #    cat_1 = concatenate([x1, x2], cat_dim=1)
    #    reshape_2 = reshape(cat_1, [batch, 4, 6])
    #    x3 = tensor([batch, 2, 6])
    #    cat_2 = concatenate([reshape_2, x3], cat_dim=1)
    # Basically, we cannot reshape either x1 or x2 to a shape while
    # keep cat_dim = 1, i.e. we cannot form a shape [batch, -1, 6] from
    # either [batch, 16] or [batch, 8].
    new_view_output_shapes = []
    view_op_output = view_op._attrs["outputs"][0]
    original_view_shape = view_op_output.shape()
    for input_idx, first_cat_input in enumerate(first_cat._attrs["inputs"]):
        input_view_shape = _make_input_view_shape(
            first_cat_input, original_view_shape, cat_dim, input_idx
        )
        if input_view_shape is None:
            return False
        new_view_output_shapes.append(input_view_shape)
    # Now we start modifying the graph.
    # make a new output tensor for the first cat
    new_first_cat_output = Tensor(
        original_view_shape,
        first_cat_output._attrs["name"],
        dtype=first_cat_output.dtype(),
    )
    transform_utils.replace_tensor(first_cat_output, new_first_cat_output)
    first_cat._attrs["outputs"][0] = new_first_cat_output
    new_first_cat_output._attrs["src_ops"].add(first_cat)

    for dst_op in new_first_cat_output._attrs["dst_ops"]:
        dst_op_type = dst_op._attrs["op"]
        if dst_op_type in _SUPPORTED_VIEW_OPS:
            # we've ensured all view ops have the same output shape before entering
            # this function, so it's safe to remove the old view ops
            transform_utils.remove_view_op_from_sorted_graph(dst_op)
        else:
            # we need to place a view op as we've changed the concat's output shape
            new_view_output = ops.reshape()(
                new_first_cat_output, first_cat_output.shape()
            )
            transform_utils.replace_tensor_for_op(
                dst_op, new_first_cat_output, new_view_output
            )

    # make a new view op for each first_cat's original input and place it between
    # the original input and the first cat
    new_first_cat_inputs = []
    # The same tensor may be used multiple times by the first cat.
    # We don't want to make one view op for each use, because it would
    # prevent us from propagating those view ops to an upper level.
    first_cat_input_to_view_output = {}
    for first_cat_input, input_view_shape in zip(
        first_cat._attrs["inputs"], new_view_output_shapes
    ):
        new_view_output = first_cat_input_to_view_output.get(first_cat_input, None)
        if new_view_output is None:
            new_view_op = type(view_op)(**view_op._get_op_attributes())
            new_view_output = _call_view_op(
                new_view_op, input_view_shape, first_cat_input
            )
            first_cat_input_to_view_output[first_cat_input] = new_view_output
            new_view_output._attrs["dst_ops"].add(first_cat)
            first_cat_input._attrs["dst_ops"].remove(first_cat)
        new_first_cat_inputs.append(new_view_output)
    first_cat._attrs["inputs"] = new_first_cat_inputs
    first_cat._attrs["original_inputs"] = list(new_first_cat_inputs)
    first_cat._attrs["input_accessors"] = [
        TensorAccessor(inp) for inp in new_first_cat_inputs
    ]
    return True


def _is_valid_cat_op(cat: Operator) -> bool:
    """
    Return true if the cat op is valid for moving the view op.
    """
    if cat._attrs["op"] != "concatenate":
        return False
    # skip if the cat has any fused strided op
    if any(mask is False for mask in cat._attrs["input_masks"]):
        return False
    # If cat carries strided input_accessors or fused view ops, we skip it
    if "input_accessors" in cat._attrs:
        if any(
            input_accessor.stride_dim is not None
            or input_accessor.actual_shapes is not None
            for input_accessor in cat._attrs["input_accessors"]
        ):
            return False
    return True


def _get_valid_view_op_and_second_cat(
    view_ops: List[Operator],
) -> Tuple[Operator, Operator]:
    """
    Return the view op and the second cat if we can find such a pair
    """
    view_op = None
    second_cat = None
    for a_view_op in view_ops:
        view_op_output = a_view_op._attrs["outputs"][0]
        next_next_ops = view_op_output._attrs["dst_ops"]
        next_concats = [n for n in next_next_ops if n._attrs["op"] == "concatenate"]
        # only allow a single concat in the view_op's dst_ops
        if len(next_concats) != 1:
            continue
        if _is_valid_cat_op(next_concats[0]):
            view_op = a_view_op
            second_cat = next_concats[0]
            break
    return (view_op, second_cat)


def _move_view_op_before_concat(
    sorted_graph: List[Tensor],
) -> Tuple[bool, List[Tensor]]:
    """
    Return a tuple of (bool, List[Tensor]), where True indicates the
    graph has been successfully changed.
    """
    changed = False
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) == 0:
            continue
        first_cat = list(src_ops)[0]
        if not _is_valid_cat_op(first_cat):
            continue
        first_cat_outputs = first_cat._attrs["outputs"]
        if len(first_cat_outputs) != 1:
            continue
        first_cat_output = first_cat_outputs[0]
        # If the first cat is a graph output, we cannot fuse it
        if first_cat_output._attrs["is_output"]:
            continue
        next_ops = first_cat_output._attrs["dst_ops"]
        if len(next_ops) == 0:
            continue
        # skip cases where the first cat op is directly connected with another cat op,
        # because moving a view op between other two cat ops would insert a view op
        # between the directly-connected cat ops. The transformed graph would contain
        # a valid rewrite pattern which could trigger another re-write, and so on.
        # Consequently, we would end up with an infinite rewriting loop, e.g.
        # cat1 + reshape + cat2, cat1 + cat3 => cat1 + cat2, cat1 + reshape + cat3 =>
        # cat1 + reshape + cat2, cat1 + cat3 => ...
        concat_ops = [op for op in next_ops if op._attrs["op"] == "concatenate"]
        if len(concat_ops) > 0:
            continue
        view_ops = [op for op in next_ops if op._attrs["op"] in _SUPPORTED_VIEW_OPS]
        # skip if none of the next ops is one of the supported view ops
        if len(view_ops) == 0:
            continue
        a_view_op = view_ops[0]
        view_output_shape = a_view_op._attrs["outputs"][0].shape()
        # handle a special case where the all view_ops have the same output shape
        if len(view_ops) > 1 and not all(
            shape_utils.is_same_shape(
                vop._attrs["outputs"][0].shape(), view_output_shape
            )
            for vop in view_ops
        ):
            continue
        if any(vop._attrs["outputs"][0]._attrs["is_output"] for vop in view_ops):
            continue
        view_op, second_cat = _get_valid_view_op_and_second_cat(view_ops)
        if second_cat is None:
            continue
        if _try_move_view_op(first_cat, second_cat, view_op):
            changed = True
    return (changed, sorted_graph)


def move_view_op_before_concat(
    sorted_graph: List[Tensor], wordir: str = None
) -> List[Tensor]:
    """
    This transformation turns "cat + view_op + cat" into "view_op + cat + cat".
    The yielded pattern may be optimized further by the transform_memory_ops pass.
    Note that this pass must be invoked before transform_strided_op_and_view_op
    and transform_strided_ops.
    """
    changed = True
    while changed:
        changed, sorted_graph = _move_view_op_before_concat(sorted_graph)
        if changed:
            sorted_graph = toposort(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)
