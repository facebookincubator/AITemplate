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
Remove no-ops from the graph.

This is a bit different from remove_unused_ops. That pass
is based on the graph structure - it removes ops tha are not
connected to the src_ops of any tensor. This pass, on the other
hand, removes things which are logically no-ops, like expands
with no expanded dims.

The reason it's not combined with removed_unused_ops is that
many of the passes in this file will want to call sanitize_sorted_graph,
but sanitize_sorted_graph calls remove_unused_ops.

Also, even if the passes in this file avoided sanitize_sorted_graph,
many other unrelated passes use sanitize_sorted_graph. We don't need to
call the passes in this file more than once.
"""
from typing import List

from aitemplate.compiler.base import IntImm, IntVar, JaggedIntVar, Operator, Tensor
from aitemplate.compiler.ops.tensor.expand import ExpandDimensionType

from aitemplate.compiler.transform import transform_utils

from aitemplate.utils import graph_utils, shape_utils
from aitemplate.utils.shape_utils import is_singleton_dimension


def _remove_id_ops(sorted_graph: List[Tensor]) -> List[Tensor]:
    """Remove identity ops."""
    ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in ops:
        if op._attrs["op"] != "identity":
            continue

        inputs = op._attrs["inputs"]
        assert len(inputs) == 1, "identity must only have 1 input"

        outputs = op._attrs["outputs"]
        identity_output = outputs[0]
        assert len(inputs) == 1, "identity must only have 1 output"

        # skip a very special case where id takes an input and produces an output
        if identity_output._attrs["is_output"] and inputs[0]._attrs["is_input"]:
            continue

        transform_utils.remove_single_tensor_op_from_sorted_graph(op)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _remove_no_op_concats(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Remove no-op concats from the graph. A no-op concat is where the output
    tensor is exactly the same as the input tensor(s) and it isn't the model output.
    This is the case when:
    1. There is a single input tensor.
    2. There is a single non-empty input tensor and the remaining input tensors
    are empty.

    x = Tensor(shape=[7])
    empty1 = Tensor(shape=[0], value=[])
    empty2 = Tensor(shape=[0], value=[])

    y1 = ops.concatenate([x])                   # Case 1
    y2 = ops.concatenate([empty1])              # Case 1
    y2 = ops.concatenate([empty1, x, empty2])   # Case 2
    """

    def is_dim_gt_zero(dim):
        if isinstance(dim, IntImm):
            return dim.value() > 0
        elif isinstance(dim, IntVar):
            return dim.lower_bound() > 0

    ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in ops:
        if op._attrs["op"] != "concatenate":
            continue

        inputs = op._attrs["inputs"]
        assert len(inputs) >= 1, "concat must have at least 1 input"

        outputs = op._attrs["outputs"]
        concat_output = outputs[0]
        assert len(outputs) == 1, "concat must have a single output"

        # Assumes non-empty tensors have non-zero dimensions.
        # And empty tensors have dimensions of size 0.
        is_input_non_empty = [
            all(is_dim_gt_zero(dim) for dim in tensor.shape()) for tensor in inputs
        ]
        n_non_empty = sum(is_input_non_empty)
        if len(inputs) > 1 and n_non_empty > 1 or outputs[0]._attrs["is_output"]:
            continue

        idx = is_input_non_empty.index(True) if n_non_empty == 1 else 0
        concat_input = inputs[idx]
        for dst_op in concat_output.dst_ops():
            transform_utils.replace_tensor_for_op(dst_op, concat_output, concat_input)
        transform_utils.remove_tensor_from_sorted_graph(concat_output)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _remove_no_op_dynamic_slices(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Remove any no-op slices from the graph. A no-op slice is when the input tensor
    and output tensor are exactly the same. This happens when the start indices
    and end indices cover the entire dimension length.

    x = Tensor([1, 2, 3])
    y = x[:]

    xx = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    yy = xx[0:2, -4:4]
    """

    ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in ops:
        if op._attrs["op"] != "dynamic_slice":
            continue

        inputs = op._attrs["inputs"]
        assert len(inputs) == 1, "dynamic_slice must only have 1 input"

        outputs = op._attrs["outputs"]
        assert len(inputs) == 1, "dynamic_slice must only have 1 output"

        slice_input, slice_output = inputs[0], outputs[0]
        if (
            not shape_utils.is_same_shape(slice_input.shape(), slice_output.shape())
            or slice_output._attrs["is_output"]
        ):
            continue

        for dst_op in slice_output.dst_ops():
            transform_utils.replace_tensor_for_op(dst_op, slice_output, slice_input)
        transform_utils.remove_tensor_from_sorted_graph(slice_output)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _remove_no_op_splits(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Remove any no-op split from the graph where the input tensor is non-jagged.
    A no-op split is where the input tensor isn't divided into multiple parts.
    This happens when the split_size_or_sections argument is:
    1. an integer representing the length of the dimension indicated by dim
    2. a singleton list containing the length of the dimension indicated by dim.

    x = Tensor([1, 2, 3])
    y1 = split(x, split_size_or_sections=3, dim=0)  # Case 1
    y2 = split(x, split_size_or_sections=[3], dim=0)   # Case 2

    xx = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    yy1 = split(xx, split_size_or_sections=2, dim=0)  # Case 1
    yy2 = split(xx, split_size_or_sections=4, dim=1)  # Case 1
    yy3 = split(xx, split_size_or_sections=[2], dim=0)  # Case 2
    yy4 = split(xx, split_size_or_sections=[4], dim=1)  # Case 2
    """

    ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in ops:
        if op._attrs["op"] != "split":
            continue

        inputs = op._attrs["inputs"]
        assert len(inputs) == 1, "split must only have 1 input"

        outputs = op._attrs["outputs"]
        assert len(inputs) >= 1, "split must have at least 1 output"

        split_dim = op._attrs["split_dim"]
        split_input, split_output = inputs[0], outputs[0]
        input_split_dim_len, output_split_dim_len = (
            split_input._attrs["shape"][split_dim],
            split_output._attrs["shape"][split_dim],
        )

        # No-op splits must have one output, and the input and output shapes
        # must match along split_dim. We ignore no-op splits that are outputs.
        if (
            len(outputs) > 1
            or input_split_dim_len != output_split_dim_len
            or outputs[0]._attrs["is_output"]
        ):
            continue

        # Delete the split output in the graph.
        for dst_op in list(split_output.dst_ops()):
            transform_utils.replace_tensor_for_op(dst_op, split_output, split_input)

        transform_utils.remove_tensor_from_sorted_graph(split_output)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _remove_no_op_expands(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Remove no-op expands from the graph. A no-op expand is one
    that doesn't expand any singleton dimensions to values greater
    than one.

    x = Tensor([1, 2, 3])
    y1 = ops.expand()(x, [-1, -1, -1])  # no-op
    y2 = ops.expand()(x, [1, 2, -1])  # no-op
    """
    ops = graph_utils.get_sorted_ops(sorted_graph)
    for op in ops:
        if op._attrs["op"] != "expand":
            continue

        outputs = op._attrs["outputs"]
        assert len(outputs) == 1, "expand must only have 1 output"
        expand_output = outputs[0]

        if expand_output._attrs["is_output"]:
            continue

        inputs = op._attrs["inputs"]
        assert len(inputs) >= 1, "expand must have at least 1 input"
        expand_input = inputs[0]

        assert len(op._attrs["dim_types"]) == len(
            expand_output._attrs["shape"]
        ), "expand must have dim_type for every output dimension"

        # If we just keep every dimension as-is, it is a no-op
        if any(dt != ExpandDimensionType.KEEP_DIM for dt in op._attrs["dim_types"]):
            continue

        # This expand is a no-op, so we know that these shapes should
        # be the same. However, the shape inference system may not be aware
        # of that due to different IntVar names.
        expand_input._attrs["shape"] = expand_output._attrs["shape"]
        for dst in list(expand_output.dst_ops()):
            transform_utils.replace_tensor_for_op(dst, expand_output, expand_input)

        transform_utils.remove_tensor_from_sorted_graph(expand_output)

    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _fuse_expand_elementwise(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Eliminate expand ops that occur before elementwise when broadcasting
    in elementwise can handle the unexpanded input.

    Example:
    x = Tensor([1, 2, 3])
    y = Tensor([3, 2, 3])
    z = ops.elementwise(FuncEnum.ADD)(ops.expand()(x, [3, 2, 3]), y)

    The expand here is not required because elementwise broadcasting will just
    do the right thing.

    Note that this must occur before any pass that fuses elementwise into
    other ops.
    """

    def _is_compatible_with_broadcasting(
        expand_output_dim: IntVar, elementwise_input_dim: IntVar
    ) -> bool:
        return expand_output_dim == elementwise_input_dim or is_singleton_dimension(
            expand_output_dim
        )

    def _replace_jagged_int_var(shape: List[IntVar]):
        """
        If shape[0] is a JaggedIntVar, replace it with
        the corresponding maximum dense shape.
        """
        if shape and isinstance(shape[0], JaggedIntVar):
            return shape[0].get_max_dense_shape() + shape[1:]
        return shape

    for op in graph_utils.get_sorted_ops(sorted_graph):
        if op._attrs["op"] != "expand":
            continue

        outputs = op._attrs["outputs"]
        assert len(outputs) == 1, "expand must only have 1 output"
        expand_output = outputs[0]

        if expand_output._attrs["is_output"]:
            continue

        expand_output_shape = _replace_jagged_int_var(expand_output._attrs["shape"])

        def _can_fuse_with(dst_op: Operator) -> bool:
            if dst_op._attrs["op"] != "elementwise":
                return False

            for elementwise_input in dst_op._attrs["inputs"]:
                if elementwise_input is expand_output:
                    continue

                elementwise_input_shape = _replace_jagged_int_var(
                    elementwise_input._attrs["shape"]
                )

                if not all(
                    _is_compatible_with_broadcasting(dim_a, dim_b)
                    for dim_a, dim_b in zip(
                        expand_output_shape,
                        elementwise_input_shape,
                    )
                ):
                    return False
            return True

        if not all(_can_fuse_with(dst) for dst in expand_output._attrs["dst_ops"]):
            continue

        inputs = op._attrs["inputs"]
        assert len(inputs) >= 1, "expand must have at least 1 input"
        expand_input = inputs[0]

        for dst in list(expand_output.dst_ops()):
            transform_utils.replace_tensor_for_op(dst, expand_output, expand_input)

        transform_utils.remove_tensor_from_sorted_graph(expand_output)


def remove_no_ops(sorted_graph: List[Tensor]) -> List[Tensor]:
    """Remove no-ops from the graph.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph

    Returns
    -------
    List[Tensor]
        Graph after remove no-ops
    """
    passes = [
        _remove_id_ops,
        _remove_no_op_concats,
        _remove_no_op_dynamic_slices,
        _remove_no_op_splits,
        _remove_no_op_expands,
        _fuse_expand_elementwise,
    ]
    for f_pass in passes:
        sorted_graph = f_pass(sorted_graph)
    return sorted_graph
