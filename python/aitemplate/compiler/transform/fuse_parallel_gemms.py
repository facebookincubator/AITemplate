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
Fuse parallel gemms into bmm op.
"""

from typing import Callable, List, Tuple

from aitemplate.compiler import ops
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.ops.gemm_universal.gemm_common import default_align_ab
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.compiler.transform.transform_strided_ops import _is_supported_op

from aitemplate.utils import graph_utils
from aitemplate.utils.shape_utils import is_static_dimension


def _is_same_shape(gemm_op1: Operator, gemm_op2: Operator) -> bool:
    inputs1 = gemm_op1._attrs["inputs"]
    inputs2 = gemm_op2._attrs["inputs"]
    if len(inputs1) != len(inputs2):
        return False
    for input1, input2 in zip(inputs1, inputs2):
        if input1._rank() != input2._rank():
            return False
        for dim1, dim2 in zip(input1.shape(), input2.shape()):
            if dim1 != dim2:
                return False
    return True


def _is_valid_gemm_op(tensor: Tensor, f_check_src_op: Callable) -> bool:
    """check if the src op of tensor is a valid gemm op for parallel fusion.

    Args:
        tensor (Tensor): the output tensor of the gemm op
        f_check_src_op (Callable): a function to check if the src op of the
        input to gemm op is valid for fusion

    Returns:
        bool: True if src_op of tensor is a valid gemm op
    """
    if len(tensor.dst_ops()) != 1 or len(tensor.src_ops()) != 1:
        return False

    # Don't fuse if tensor is an output tensor
    if tensor._attrs["is_output"]:
        return False

    gemm_op = list(tensor.src_ops())[0]
    if gemm_op._attrs["op"] != "gemm_rcr_bias":
        return False

    gemm_input, weight, bias = gemm_op._attrs["inputs"]

    # check gemm weight/bias is available for constant folding
    if not transform_utils.can_be_constant_folded([weight, bias]):
        return False

    if len(gemm_input.dst_ops()) != 1 or len(gemm_input.src_ops()) != 1:
        return False

    # perm102_bmm only supports 3D input, 3D weight, 2D bias
    if gemm_input._rank() != 2 or weight._rank() != 2 or bias._rank() != 1:
        return False

    if not is_static_dimension(gemm_input.shape(), 1):
        return False

    if not is_static_dimension(weight.shape(), 0) or not is_static_dimension(
        weight.shape(), 1
    ):
        return False
    if not is_static_dimension(bias.shape(), 0):
        return False

    src_op = list(gemm_input.src_ops())[0]

    # the new cat must be eliminated with TensorAccessor
    if not f_check_src_op(src_op):
        return False
    return True


def _get_row_length(cat_input: Tensor):
    shape = cat_input.shape()
    return shape[-1].value()


# find groups of parallel gemm ops with identical shapes
def _find_parallel_gemm_ops(
    cat_inputs: List[Tensor], f_check_src_op: Callable
) -> List[Tuple[List[Operator], int]]:
    all_groups = []
    gemm_ops = []

    def add_gemm_groups(gemm_ops):
        if len(gemm_ops) >= 2:
            all_groups.append((gemm_ops.copy()))

    for cat_input in cat_inputs:
        if not _is_valid_gemm_op(cat_input, f_check_src_op):
            add_gemm_groups(gemm_ops)
            gemm_ops.clear()
        else:
            gemm_op = list(cat_input.src_ops())[0]
            if len(gemm_ops) == 0:
                gemm_ops.append(gemm_op)
                continue
            if _is_same_shape(gemm_ops[-1], gemm_op):
                gemm_ops.append(gemm_op)
            else:
                # start new group when the gemm shape is different
                add_gemm_groups(gemm_ops)
                gemm_ops.clear()
                gemm_ops.append(gemm_op)

    # handle last group
    add_gemm_groups(gemm_ops)
    return all_groups


def _group_gemm_inputs(gemm_ops: List[Operator]) -> Tuple[List[Tensor]]:
    inputs = []
    weights = []
    bias = []
    for gemm_op in gemm_ops:
        gemm_inputs = gemm_op._attrs["inputs"]
        assert len(gemm_inputs) == 3
        inputs.append(gemm_inputs[0])
        weights.append(gemm_inputs[1])
        bias.append(gemm_inputs[2])
    return inputs, weights, bias


def _clear_gemm_inputs_dst_ops(gemm_ops: List[Operator]):
    for gemm_op in gemm_ops:
        gemm_inputs = gemm_op._attrs["inputs"]
        for input in gemm_inputs:
            input.dst_ops().clear()


def _get_gemm_output_idx_in_cat_inputs(gemm_op, cat_op):
    gemm_outputs = gemm_op._attrs["outputs"]
    assert len(gemm_outputs) == 1
    gemm_output = gemm_outputs[0]
    idx = cat_op.get_tensor_index(gemm_output)
    return idx


def _merge_parallel_gemm_concat(
    gemm_ops: List[Operator], cat_op: Operator, sorted_graph: List[Tensor]
):
    """merge parallel gemm ops and the following concat op together"""
    # clear gemm_inputs dst_ops
    _clear_gemm_inputs_dst_ops(gemm_ops)

    inputs, weights, bias = _group_gemm_inputs(gemm_ops)

    n, k = weights[0].shape()[0].value(), weights[0].shape()[1].value()
    b = len(weights)

    dtype = inputs[0].dtype()
    rcr_align = default_align_ab(k, k, dtype)
    rrr_align = default_align_ab(k, n, dtype)

    use_rcr = rcr_align > rrr_align

    # create new subgraph
    bmm_input_cat = ops.concatenate()(inputs, dim=-1)
    bmm_input = ops.reshape()(bmm_input_cat, [-1, b, k])

    bmm_weight_cat = ops.concatenate()(weights, dim=0)
    bmm_weight_reshape = ops.reshape()(bmm_weight_cat, [b, n, k])
    bmm_weight = bmm_weight_reshape if use_rcr else ops.permute021()(bmm_weight_reshape)

    bmm_bias_cat = ops.concatenate()(bias, dim=0)
    bmm_bias = ops.reshape()(bmm_bias_cat, [b, n])

    if use_rcr:
        bmm = ops.perm102_bmm_rcr_bias()(bmm_input, bmm_weight, bmm_bias)
    else:
        bmm = ops.perm102_bmm_rrr_bias()(bmm_input, bmm_weight, bmm_bias)
    bmm_reshape = ops.reshape()(bmm, [-1, b * n])

    num_cat_inputs = len(cat_op._attrs["inputs"])
    cat_output = cat_op._attrs["outputs"][0]
    if len(gemm_ops) == num_cat_inputs:
        # fuse with concat op completely
        transform_utils.replace_tensor(cat_output, bmm_reshape)

        # if cat_output was the only output of the graph, we must
        # append the new graph output to the graph
        sorted_graph.append(bmm_reshape)

    else:
        # bmm_reshape now replaces num_cat_inputs cat inputs
        begin_idx = _get_gemm_output_idx_in_cat_inputs(gemm_ops[0], cat_op)
        end_idx = _get_gemm_output_idx_in_cat_inputs(gemm_ops[-1], cat_op)

        old_inputs = cat_op._attrs["inputs"]
        new_inputs = old_inputs[:begin_idx] + [bmm_reshape] + old_inputs[end_idx + 1 :]

        assert all(
            cat_op._attrs["input_masks"]
        ), "The input_pasts of cat_op must be all True"

        cat_op._attrs["inputs"] = new_inputs
        cat_op._attrs["input_accessors"] = [TensorAccessor(t) for t in new_inputs]
        cat_op._attrs["original_inputs"] = list(new_inputs)
        cat_op._attrs["input_masks"] = [True] * len(new_inputs)

        bmm_reshape._attrs["dst_ops"].add(cat_op)

    for gemm_op in gemm_ops:
        gemm_outputs = gemm_op._attrs["outputs"]
        assert len(gemm_outputs) == 1
        transform_utils.remove_tensor_from_sorted_graph(gemm_outputs[0])


def _check_cat_op(op: Operator) -> bool:
    cat_inputs = op._attrs["inputs"]
    if len(cat_inputs) <= 1:
        return False
    rank = cat_inputs[0]._rank()
    if op._attrs["concat_dim"] != rank - 1:
        return False
    return True


def _fuse_parallel_gemm_concat(sorted_graph: List[Tensor]) -> List[Tensor]:
    """This pass fuses patterns like
    # x1: [m, k], w1: [n, k], b1: [n]
    y1 = gemm_rcr_bias()(x1, w1, b1)
    y2 = gemm_rcr_bias()(x2, w2, b1)
    y3 = concatenate()([x1, x2], dim=-1)

    into:
    # x: [m, b, k], w: [b, k, n], b: [b, n]
    x = concatenate()([x1, x2], dim=-1).reshape([-1, b, n])
    w = concatenate()([w1, w2], dim=-1).reshape([b, n, k]).permute([0, 2, 1])
    b =  concatenate()([b1, b2], dim=-1).reshape([b, n])
    y3 = perm102_bmm_rrr_bias()(x, w, b).reshape([-1, b * n])

    after constant folding:
    x = concatenate()([x1, x2], dim=-1).reshape([-1, b, n])
    y3 = perm102_bmm_rrr_bias()(x, w, b).reshape([-1, b * n])

    If rcr layout has better alignment than rrr, it will tranform the graph into

    # x: [m, b, k], w: [b, n, k], b: [b, n]
    x = concatenate()([x1, x2], dim=-1).reshape([-1, b, n])
    y3 = perm102_bmm_rcr_bias()(x, w, b).reshape([-1, b * n])

    If there are other inputs to the concat op, such as

    y1 = gemm_rcr_bias()(x1, w1, b1)
    y2 = gemm_rcr_bias()(x2, w2, b1)
    y3 = concatenate()([y1, y2, x3, x4], dim=-1)

    The graph is transformed into
    x = concatenate()([x1, x2], dim=-1).reshape([-1, b, n])
    y3 = perm102_bmm_rrr_bias()(x, w, b).reshape([-1, b * n])
    y4 = concatenate()([y3, x3, x4], dim=-1)

    y3 will write into the y4 directly through concat fusion.

    For w and b, we rely on constant folding to preprocess them.
    For the extra concat op to cat x1 and x2 together, we require that the ops
    that produce x1 and x2 write directly to the output of concat.
    It is required that all the gemm ops have the same problem sizes and layouts.

    On graph pass ordering, we need to make sure this pass runs before
    any other pass that modifies gemm and concat input/output TensorAccessors.

    For odd k/n, we rely on apply_padding pass to add padding to X and W.
    The overall perf may be better or worse depending on problem sizes.

    Args:
        sorted_graph (List[Tensor]): a sorted list of tensors

    Returns:
        List[Tensor]: the transformed graph with all ops sorted
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    fusion_groups = []
    for op in sorted_ops:
        # check cat op
        if op._attrs["op"] != "concatenate":
            continue
        if not _check_cat_op(op):
            continue

        cat_inputs = op._attrs["inputs"]

        all_groups = _find_parallel_gemm_ops(cat_inputs, _is_supported_op)

        for gemm_ops in all_groups:
            # TODO: 2 is arbitrarily chosen. More benchmarks with real models
            # are needed to find the best criteria. If gemms are not fused here,
            # they can be grouped into group gemms with other gemms.
            if len(gemm_ops) >= 2:
                fusion_groups.append([gemm_ops, op])

    for gemm_ops, cat_op in fusion_groups:
        _merge_parallel_gemm_concat(gemm_ops, cat_op, sorted_graph)
    sorted_graph = toposort(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def _is_split_op(op: Operator) -> bool:
    op_type = op._attrs["op"]
    if op_type != "split":
        return False
    split_dim = op._attrs["split_dim"]
    inputs = op._attrs["inputs"]
    if len(inputs) == 0:
        return False
    if split_dim != inputs[0]._rank() - 1:
        return False
    return True


def _from_same_src_op(gemm_ops: List[Operator], op_type: str) -> bool:
    """
    Check that the first input of all the ops in gemm_ops come from the same exact op.
    Returns true if they all come from the same op, and false otherwise.
    """
    if len(gemm_ops) <= 1:
        return True
    src_ops = list(gemm_ops[0]._attrs["inputs"][0].src_ops())
    if len(src_ops) != 1:
        return False
    src_op = src_ops[0]
    if src_op._attrs["op"] != op_type:
        return False
    for gemm_op in gemm_ops[1:]:
        src_ops = gemm_op._attrs["inputs"][0].src_ops()
        if len(src_ops) != 1:
            return False
        if src_op not in src_ops:
            return False
    return True


def _fuse_split_parallel_gemm_concat(sorted_graph: List[Tensor]) -> List[Tensor]:
    """This pass fuses patterns like
    # x: [m, b * k], w1: [n, k], b1: [n]
    x1, x2 = split()(x, k, dim=-1)
    y1 = gemm_rcr_bias()(x1, w1, b1)
    y2 = gemm_rcr_bias()(x2, w2, b1)
    y3 = concatenate()([x1, x2], dim=-1)

    first into:
    # x: [m, b, k], w: [b, k, n], b: [b, n]
    x1, x2 = split()(x, k, dim=-1)
    x = concatenate()([x1, x2], dim=-1).reshape([-1, b, n])
    w = concatenate()([w1, w2], dim=-1).reshape([b, n, k]).permute([0, 2, 1])
    b =  concatenate()([b1, b2], dim=-1).reshape([b, n])
    y3 = perm102_bmm_rrr_bias()(x, w, b).reshape([-1, b * n])

    after transform_memory_ops pass:
    x = x.reshape([-1, b, n])
    w = concatenate()([w1, w2], dim=-1).reshape([b, n, k]).permute([0, 2, 1])
    b =  concatenate()([b1, b2], dim=-1).reshape([b, n])
    y3 = perm102_bmm_rrr_bias()(x, w, b).reshape([-1, b * n])

    after constant folding:
    x = x.reshape([-1, b, n])
    y3 = perm102_bmm_rrr_bias()(x, w, b).reshape([-1, b * n])

    If rcr layout has better alignment than rrr, it will tranform the graph into

    # x: [m, b, k], w: [b, n, k], b: [b, n]
    x = concatenate()([x1, x2], dim=-1).reshape([-1, b, n])
    y3 = perm102_bmm_rcr_bias()(x, w, b).reshape([-1, b * n])

    If there are other inputs to the concat op, such as

    x1, x2 = split()(x, k, dim=-1)
    y1 = gemm_rcr_bias()(x1, w1, b1)
    y2 = gemm_rcr_bias()(x2, w2, b1)
    y3 = concatenate()([y1, y2, x3, x4], dim=-1)

    The graph is transformed into
    x = x.reshape([-1, b, n])
    y3 = perm102_bmm_rrr_bias()(x, w, b).reshape([-1, b * n])
    y4 = concatenate()([y3, x3, x4], dim=-1)

    y3 will write into the y4 directly through concat fusion.

    For w and b, we rely on constant folding to preprocess them.
    It is required that all the gemm ops have the same problem sizes and layouts.
    We also check that all the gemm inputs come from the same split op.

    Args:
        sorted_graph (List[Tensor]): a sorted list of tensors

    Returns:
        List[Tensor]: the transformed graph with all ops sorted
    """
    sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
    fusion_groups = []
    for op in sorted_ops:
        if op._attrs["op"] != "concatenate":
            continue
        if not _check_cat_op(op):
            continue

        cat_inputs = op._attrs["inputs"]

        all_groups = _find_parallel_gemm_ops(cat_inputs, _is_split_op)

        for gemm_ops in all_groups:
            if not _from_same_src_op(gemm_ops, "split"):
                continue
            fusion_groups.append([gemm_ops, op])

    for gemm_ops, cat_op in fusion_groups:
        _merge_parallel_gemm_concat(gemm_ops, cat_op, sorted_graph)
    sorted_graph = toposort(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def fuse_parallel_gemms(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Fuse parallel gemms into a single gemm op.
    Currently, we only support the following patterns:

    - parallel gemm + concat
    - split->parallel gemm->concat

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph
    workdir : str, optional
        working dir, by default None

    Returns
    -------
    List[Tensor]
        Fused graph
    """
    funcs = [
        _fuse_parallel_gemm_concat,
        _fuse_split_parallel_gemm_concat,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
