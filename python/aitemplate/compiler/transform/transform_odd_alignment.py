# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Add permute for gemm/bmm if alignment is odd.
"""
from typing import Dict, List, Optional, Set, Tuple

from ..base import IntImm, Operator, Tensor
from ..ops.gemm_universal import bmm_ccr, bmm_crr, bmm_rcr, bmm_rrr
from ..ops.tensor import permute021
from .transform_utils import (
    copy_src_op_attributes,
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    remove_tensor_from_sorted_graph,
    replace_tensor,
    sanitize_sorted_graph,
)

# pylint: disable=C0103,W0612


def _extract_only_one_op(ops: Set[Optional[Operator]]) -> Optional[Operator]:
    """
    Helper function that returns the op from src_ops() or dst_ops() call.
    Return None if there's no ops or if there's more than one op.
    """
    if ops is None or len(ops) != 1:
        return None
    return list(ops)[0]


def _transform_odd_alignment(
    sorted_graph: List[Tensor],
    permutable_pairs: Dict[str, Tuple[Operator, Operator, Operator]],
) -> List[Tensor]:
    """
    Function that fuses [permute021 + bmm] into corresponding bmm op.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        AIT graph to run fusion
    permutable_pairs: Dict[str, Tuple[Operator, Operator, Operator]]
        A dictionary where key is the op that might accept permuted inputs,
        and value is a tuple of len-3 which correspond to ops (permute first input,
        permute second input, permute both first and second inputs)
    """

    new_sorted_graph = []
    permuted_inputs = {}
    for tensor in sorted_graph:
        new_sorted_graph.append(tensor)

        src_op = _extract_only_one_op(tensor._attrs["src_ops"])
        if src_op is None:
            continue

        op_type = src_op._attrs["op"]
        if op_type not in permutable_pairs:
            continue

        permute_input = [False, False]
        inputs = src_op._attrs["inputs"]
        a_shapes = inputs[0].shape()
        b_shapes = inputs[1].shape()

        if (
            isinstance(a_shapes[-1], IntImm)
            and a_shapes[-1].value() % 2 == 1
            and isinstance(a_shapes[-2], IntImm)
            and a_shapes[-2].value() % 2 == 0
        ):
            permute_input[0] = True
        if (
            isinstance(b_shapes[-1], IntImm)
            and b_shapes[-1].value() % 2 == 1
            and isinstance(b_shapes[-2], IntImm)
            and b_shapes[-2].value() % 2 == 0
        ):
            permute_input[1] = True

        # TODO: Apply check on whether input is ConstantTensor
        if not permute_input[0] and not permute_input[1]:
            continue

        new_inputs = list(inputs)
        for idx in range(2):
            if permute_input[idx]:
                if inputs[idx] in permuted_inputs:
                    permuted_input = permuted_inputs[inputs[idx]]
                else:
                    permuted_input = permute021()(inputs[idx])
                    new_sorted_graph.insert(-1, permuted_input)
                    permuted_inputs[inputs[idx]] = permuted_input
                new_inputs[idx] = permuted_input

        if permute_input[0] and permute_input[1]:
            new_tensor = permutable_pairs[op_type][2]()(*new_inputs)
        elif permute_input[0]:
            new_tensor = permutable_pairs[op_type][0]()(*new_inputs)
        elif permute_input[1]:
            new_tensor = permutable_pairs[op_type][1]()(*new_inputs)
        copy_tensor_attributes(new_tensor, tensor)
        copy_src_op_attributes(new_tensor, tensor)
        replace_tensor(tensor, new_tensor)

        remove_dst_op_from_tensor(inputs, src_op)
        remove_tensor_from_sorted_graph(tensor)

        new_sorted_graph[-1] = new_tensor

    return sanitize_sorted_graph(new_sorted_graph)


def transform_odd_alignment(sorted_graph: List[Tensor]) -> List[Tensor]:
    permutable_pairs = {
        "bmm_ccr": (bmm_rcr, bmm_crr, bmm_rrr),
        "bmm_crr": (bmm_rrr, bmm_ccr, bmm_rcr),
        "bmm_rcr": (bmm_ccr, bmm_rrr, bmm_crr),
        "bmm_rrr": (bmm_crr, bmm_rcr, bmm_ccr),
    }

    sorted_graph = _transform_odd_alignment(sorted_graph, permutable_pairs)

    return sorted_graph
