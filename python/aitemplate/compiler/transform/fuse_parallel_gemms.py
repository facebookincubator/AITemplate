# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] Fuse parallel gemms into bmm op.
"""

from typing import Callable, List, Tuple

from ...utils import graph_utils
from ...utils.shape_utils import is_static_dimension
from .. import ops
from ..base import Operator, Tensor
from ..ops.gemm_universal.gemm_common import default_align_ab
from . import transform_utils
from .toposort import toposort
from .transform_strided_ops import _is_supported_op


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
    offset = 0
    i = 0
    for i, cat_input in enumerate(cat_inputs):
        cat_input = cat_inputs[i]
        if not _is_valid_gemm_op(cat_input, f_check_src_op):
            if len(gemm_ops) >= 2:
                all_groups.append((gemm_ops.copy(), offset))
            gemm_ops.clear()
        else:
            gemm_op = list(cat_input.src_ops())[0]
            if len(gemm_ops) == 0 or _is_same_shape(gemm_ops[-1], gemm_op):
                gemm_ops.append(gemm_op)
        # update offset
        offset += _get_row_length(cat_input)

    if len(gemm_ops) == 1:
        gemm_ops.clear()
    else:
        all_groups.append((gemm_ops.copy(), offset))
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


def _merge_parallel_gemm_concat(
    gemm_ops: List[Operator], cat_op: Operator, sorted_graph: List[Tensor]
):
    """merge parallel gemm ops and the following concat op together"""
    # clear gemm_inputs dst_ops
    _clear_gemm_inputs_dst_ops(gemm_ops)

    inputs, weights, bias = _group_gemm_inputs(gemm_ops)

    n, k = weights[0].shape()[0].value(), weights[0].shape()[1].value()
    b = len(weights)

    rcr_align = default_align_ab(k, k)
    rrr_align = default_align_ab(k, n)

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

    cat_output = cat_op._attrs["outputs"][0]
    transform_utils.replace_tensor(cat_output, bmm_reshape)

    # if cat_output was the only output of the graph, we must
    # append the new graph output to the graph
    sorted_graph.append(bmm_reshape)

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

    For w and b, we rely on constant folding to preprocess them.
    For the extra concat op to cat x1 and x2 together, we require that the ops
    that produce x1 and x2 write directly to the output of concat.
    It is required that all the gemm ops have the same problem sizes and layouts
    and there is no other input to the concat op.

    On graph pass ordering, we need to make sure this pass runs before
    any other pass that modifies gemm input/output TensorAccessors.

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
        if len(all_groups) != 1:
            continue
        gemm_ops, offset = all_groups[0]

        # TODO: support partial cat fusion
        if len(gemm_ops) != len(cat_inputs):
            continue
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

    after _merge_split_and_cat pass:
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

    For w and b, we rely on constant folding to preprocess them.
    It is required that all the gemm ops have the same problem sizes and layouts
    and there is no other input to the concat op. We also check that all the gemm
    inputs come from the same split op.

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
        if len(all_groups) != 1:
            continue

        # all outputs of split must be able to be fused
        gemm_ops, _ = all_groups[0]
        if len(gemm_ops) != len(cat_inputs):
            continue
        if not _from_same_src_op(gemm_ops, "split"):
            continue

        fusion_groups.append([gemm_ops, op])

    for gemm_ops, cat_op in fusion_groups:
        _merge_parallel_gemm_concat(gemm_ops, cat_op, sorted_graph)
    sorted_graph = toposort(sorted_graph)
    return transform_utils.sanitize_sorted_graph(sorted_graph)


def fuse_parallel_gemms(sorted_graph: List[Tensor]) -> List[Tensor]:
    funcs = [
        _fuse_parallel_gemm_concat,
        _fuse_split_parallel_gemm_concat,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
