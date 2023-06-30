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
Perform transformations to fuse view ops with strided op by using TensorAccessor.
"""

from typing import List

from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.public import IntImm
from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.transform import transform_utils
from aitemplate.utils import graph_utils


_VIEW_OPS = {"reshape", "flatten", "squeeze", "unsqueeze"}


def _is_supported_strided_op(op: Operator) -> bool:
    op_kind = op._attrs["op"]
    return not op_kind.startswith("group_gemm")


def _is_supported_view_op(op: Operator, tensor: Tensor) -> bool:
    if op._attrs["op"] not in _VIEW_OPS:
        return False
    input_dynamic_dims = {
        dim
        for dim in op._attrs["inputs"][0]._attrs["shape"]
        if not isinstance(dim, IntImm)
    }
    output_dynamic_dims = {
        dim
        for dim in op._attrs["outputs"][0]._attrs["shape"]
        if not isinstance(dim, IntImm)
    }
    if input_dynamic_dims != output_dynamic_dims:
        return False
    return tensor is op._attrs["inputs"][0] or tensor is op._attrs["outputs"][0]


def _fuse_strided_op_and_view_op_single_pass(
    sorted_graph: List[Tensor],
) -> List[Tensor]:
    for tensor in sorted_graph:
        if len(tensor._attrs["src_ops"]) != 1:
            continue
        view_op = list(tensor._attrs["src_ops"])[0]
        if not _is_supported_view_op(view_op, tensor):
            continue
        view_input_tensor = view_op._attrs["inputs"][0]
        src_op = (
            list(view_input_tensor._attrs["src_ops"])[0]
            if len(view_input_tensor._attrs["src_ops"]) == 1
            else None
        )
        if (
            src_op is not None
            and len(view_input_tensor._attrs["dst_ops"]) == 1
            and "output_accessors" in src_op._attrs
            and _is_supported_strided_op(src_op)
            and not view_input_tensor._attrs["is_output"]
        ):
            found_tensor = False
            for idx, accessor in enumerate(src_op._attrs["output_accessors"]):
                if src_op._attrs["outputs"][idx] is view_input_tensor:
                    found_tensor = True
                    accessor.update_base_tensor_shape(tensor)
                    tensor._attrs["is_view_of"] = None
                    src_op._attrs["outputs"][idx] = tensor
                    tensor._attrs["src_ops"] = StableSet({src_op})
                    transform_utils.remove_tensor_from_sorted_graph(view_input_tensor)
                    break
            assert (
                found_tensor
            ), f"Cannot find view_input_tensor {view_input_tensor} from src_op outputs {src_op._attrs['outputs']}!"
        else:
            if tensor._attrs["is_output"]:
                continue
            # We have special handling for group_gemm + reshape + concat
            # in transform_strided_ops, so we skip group_gemm at the moment.
            # Otherwise, we would end up with shape mismatch due to fusing
            # the view op. We may relax this constraint if we remove the special
            # pass above.
            if src_op is not None and src_op._attrs["op"].startswith("group_gemm"):
                continue
            to_be_removed_dst_ops = set()
            for dst_op in tensor._attrs["dst_ops"]:
                if (
                    "input_accessors" not in dst_op._attrs
                    or not _is_supported_strided_op(dst_op)
                ):
                    continue
                found_tensor = False
                for idx, accessor in enumerate(dst_op._attrs["input_accessors"]):
                    if dst_op._attrs["inputs"][idx] == tensor:
                        found_tensor = True
                        accessor.update_base_tensor_shape(view_input_tensor)
                        dst_op._attrs["inputs"][idx] = view_input_tensor
                        view_input_tensor._attrs["dst_ops"].add(dst_op)
                assert (
                    found_tensor
                ), f"Cannot find tensor {tensor} from dst_op inputs {dst_op._attrs['inputs']}!"
                to_be_removed_dst_ops.add(dst_op)
            tensor._attrs["dst_ops"] = tensor._attrs["dst_ops"] - to_be_removed_dst_ops
            if len(tensor._attrs["dst_ops"]) == 0:
                view_input_tensor._attrs["dst_ops"].remove(view_op)
                transform_utils.remove_tensor_from_sorted_graph(tensor)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _fuse_strided_op_and_view_op(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    This pass fuses a view op with a strided op before or after it by using
    op._attrs["input_accessors"] or op._attrs["output_accessors"].
    It only works for ops with these fields.

    If there are multiple view ops, all these view ops will be fused into
    adjacent TensorAccessors if applicable.

    Note that are several conditions for fusion:

    1) The view op doesn't generate new dynamic dims.
    e.g. reshape(X[IntVar("batch_size"), 4], [-1, 2]) won't be fused with
    a strided op as a new dynamic dim is generated. This is because
    TensorAccessor doesn't support dynamic stride calculation for now.
    When the support is ready this condition can be removed.

    2) Some strided ops are not supported, e.g. group_gemm.
    This is because group_gemm has online_shape_inference which updates
    output tensor shapes. When this bug is fixed this condition can be
    removed.
    """

    num_ops = len(graph_utils.get_sorted_ops(sorted_graph))
    should_continue = True
    while should_continue:
        sorted_graph = _fuse_strided_op_and_view_op_single_pass(sorted_graph)
        new_num_ops = len(graph_utils.get_sorted_ops(sorted_graph))
        if num_ops == new_num_ops:
            should_continue = False
        num_ops = new_num_ops
    return sorted_graph
