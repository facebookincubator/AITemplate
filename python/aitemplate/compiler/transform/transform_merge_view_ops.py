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
This file implements a pass that merges consecutive view ops if possible.
"""
from typing import List, Set

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.utils.shape_utils import convert_shape_to_IntVarTensor


_VIEW_OPS = {"reshape", "flatten", "squeeze", "unsqueeze"}


def _is_inout(t: Tensor):
    return t._attrs["is_input"] or t._attrs["is_output"]


def _merge_view_ops_for(graph: List[Tensor], tensor: Tensor) -> List[Tensor]:
    """
    `tensor` should have exactly 1 src op, and that op must be a view op. We
    will look for view ops in the dst ops and merge them with the src view op
    by creating a new reshape op.
    """
    src_op = tensor._attrs["src_ops"][0]
    in_tensor = src_op._attrs["inputs"][0]
    dst_ops = tensor._attrs["dst_ops"]
    removed_ops: Set[Operator] = set()
    for op in dst_ops:
        if op._attrs["op"] not in _VIEW_OPS:
            continue
        out_tensor = op._attrs["outputs"][0]
        in_shape = in_tensor._attrs["shape"]
        out_shape = out_tensor._attrs["shape"]
        if out_shape == in_shape and not (
            _is_inout(in_tensor) and _is_inout(out_tensor)
        ):
            # If the shapes are identical, we can eliminate both view ops
            transform_utils.replace_tensor(out_tensor, in_tensor)
        else:
            # Otherwise, create a new reshape op to replace the two view ops
            out_shape = convert_shape_to_IntVarTensor(out_tensor)
            new_out_tensor = ops.reshape()(in_tensor, out_shape)
            if out_tensor._attrs["is_output"]:
                new_out_tensor._attrs["is_output"] = True
                new_out_tensor._attrs["name"] = out_tensor._attrs["name"]
            transform_utils.replace_tensor(out_tensor, new_out_tensor)
            graph.append(new_out_tensor)
        graph.remove(out_tensor)
        removed_ops.add(op)
    for op in removed_ops:
        transform_utils.remove_view_op_from_sorted_graph(op)
    return graph


def merge_view_ops(sorted_graph: List[Tensor], workdir: str = None) -> List[Tensor]:
    """
    Merge consecutive view ops.
    """
    changed = False
    # Find pairs of consecutive view ops and merge them, iterating to a
    # fixpoint.
    # TODO: Instead of merging pairs of view ops, we should look for entire
    # chains of view ops and merge them all at once.
    while True:
        for tensor in sorted_graph:
            src_ops = tensor._attrs["src_ops"]
            if len(src_ops) != 1:
                continue
            src_op = list(src_ops)[0]
            if src_op._attrs["op"] not in _VIEW_OPS:
                continue
            dst_ops = tensor._attrs["dst_ops"]
            if any(op._attrs["op"] in _VIEW_OPS for op in dst_ops):
                # NOTE: _merge_view_ops_for does *not* return a sorted graph
                sorted_graph = _merge_view_ops_for(sorted_graph, tensor)
                changed = True
                break
        else:
            break

    if changed:
        # Prune tensors that may have become unused after view op merging
        sorted_graph = toposort([t for t in sorted_graph if t._attrs["is_output"]])
        return transform_utils.sanitize_sorted_graph(toposort(sorted_graph))
    return sorted_graph
