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

from aitemplate.compiler.base import IntVar, JaggedIntVar, Operator, Tensor
from aitemplate.compiler.ops.tensor.expand import ExpandDimensionType

from aitemplate.compiler.transform import transform_utils

from aitemplate.utils import graph_utils
from aitemplate.utils.shape_utils import is_singleton_dimension


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
        _remove_no_op_expands,
        _fuse_expand_elementwise,
    ]
    for f_pass in passes:
        sorted_graph = f_pass(sorted_graph)
    return sorted_graph
