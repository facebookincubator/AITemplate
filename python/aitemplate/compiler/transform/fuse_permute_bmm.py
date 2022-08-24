# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Perform fusions for permute+bmm operators.
"""
from typing import Callable, List, Optional, Set, Tuple, Type, Union

from .. import ops
from ..base import IntImm, Operator, Tensor
from ..ops.gemm_universal import (
    bmm_ccr,
    bmm_crr,
    bmm_rcr,
    bmm_rrr,
    gemm_rcr,
    gemm_rcr_bias,
    gemm_rrr,
    gemm_rrr_bias,
)
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


def _extract_only_one_op(ops: Set[Union[None, Operator]]) -> Union[None, Operator]:
    """
    Helper function that returns the op from src_ops() or dst_ops() call.
    Return None if there's no ops or if there's more than one op.
    """
    if ops is None or len(ops) != 1:
        return None
    return list(ops)[0]


def _try_extract_one_mm_op(ops: Set[Union[None, Operator]]) -> Union[None, Operator]:
    """
    Helper function that returns the matmul op from src_ops() or dst_ops() call.
    Return None if there's no bmm ops
    """
    if ops is None:
        return None

    for op in ops:
        if op._attrs["op"].startswith("bmm") or op._attrs["op"].startswith("gemm"):
            return op

    return None


def _fuse_permute_bmm_ops(
    sorted_graph: List[Tensor],
    source: List[Type[Operator]],
    targets: List[Union[None, Type[Operator]]],
    condition: Optional[Callable],
) -> Tuple[bool, List[Tensor]]:
    """
    Function that fuses [permute021 + bmm] into corresponding bmm op.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        AIT graph to run fusion
    source: List[Type[Operator]]
        Combination of permute+bmm ops to be fused.
        This should be of len-2
    targets: List[Type[Operator]]
        To be fused bmm that matches the source.
        This should be of len 2, which corresponds to the operator that does
        permute A and permute B respectively
    condition: Optional[Callable]
        If not None, we apply on the gemm op to check whether it requires fusion.
    """
    assert len(source) == 2, "Source should have 2 elements, got {} instead".format(
        len(source)
    )

    new_sorted_graph = []
    fused = False
    to_replace = {}
    for tensor in sorted_graph:
        if tensor in to_replace:
            new_sorted_graph.append(to_replace[tensor])
            replace_tensor(tensor, to_replace[tensor])
            del to_replace[tensor]
            continue
        new_sorted_graph.append(tensor)

        if fused:
            continue
        if tensor._attrs["is_output"]:
            continue

        permute_op = _extract_only_one_op(tensor._attrs["src_ops"])
        bmm_op = _try_extract_one_mm_op(tensor._attrs["dst_ops"])
        if permute_op is None or bmm_op is None:
            continue

        if permute_op._attrs["op"] != source[0]()._attrs["op"]:
            continue
        if bmm_op._attrs["op"] != source[1]()._attrs["op"]:
            continue
        if condition is not None and not condition(bmm_op):
            continue

        assert len(permute_op._attrs["inputs"]) == 1
        assert len(bmm_op._attrs["outputs"]) == 1

        inputs = list(bmm_op._attrs["inputs"])
        if targets[0] is None and inputs[0] == tensor:
            continue
        if targets[1] is None and inputs[1] == tensor:
            continue

        input_tensor = permute_op._attrs["inputs"][0]
        output_tensor = bmm_op._attrs["outputs"][0]

        # TODO: Check whether the input is weight to have better compile time
        #       optimization on preprocessing of pad etc.
        permute_shape = tensor.shape()
        prepermute_shape = input_tensor.shape()

        if (
            isinstance(prepermute_shape[-1], IntImm)
            and prepermute_shape[-1].value() % 2 == 1
            and isinstance(permute_shape[-1], IntImm)
            and permute_shape[-1].value() % 2 == 0
        ):
            # We don't run the permute+bmm fusion if the permute op could
            # turn an odd alignment into even alignment.
            continue

        fused = True

        remove_dst_op_from_tensor(bmm_op._attrs["inputs"], bmm_op)

        target = None
        if inputs[0] == tensor:
            target = targets[0]
            inputs[0] = input_tensor
        elif inputs[1] == tensor:
            target = targets[1]
            inputs[1] = input_tensor
        else:
            raise RuntimeError(
                "bmm inputs are {}, not matching permute's output tensor {}".format(
                    inputs, tensor
                )
            )

        if not tensor.dst_ops():
            # Remove permute configs if this is the last bmm consuming the tensor
            remove_dst_op_from_tensor(input_tensor, permute_op)
            remove_tensor_from_sorted_graph(tensor)

        new_tensor = target()(*inputs)
        copy_tensor_attributes(new_tensor, output_tensor)
        copy_src_op_attributes(new_tensor, output_tensor)
        to_replace[output_tensor] = new_tensor

    return (fused, sanitize_sorted_graph(new_sorted_graph))


def fuse_permute_bmm(sorted_graph: List[Tensor]) -> List[Tensor]:
    def _need_broadcast_gemm(op: Operator):
        if not op._attrs["op"].startswith("gemm"):
            return False
        inputs = op._attrs["inputs"]
        return len(inputs[0].shape()) != 2 or len(inputs[1].shape()) != 2

    permute_mm_patterns = (
        ([permute021, bmm_ccr], [bmm_rcr, bmm_crr], None),
        ([permute021, bmm_crr], [bmm_rrr, bmm_ccr], None),
        ([permute021, bmm_rcr], [bmm_ccr, bmm_rrr], None),
        ([permute021, bmm_rrr], [bmm_crr, bmm_rcr], None),
        ([permute021, gemm_rcr], [bmm_ccr, bmm_rrr], _need_broadcast_gemm),
        ([permute021, gemm_rrr], [bmm_crr, bmm_rcr], _need_broadcast_gemm),
        (
            [permute021, gemm_rcr_bias],
            [ops.gemm_universal.bmm_ccr_add, ops.gemm_universal.bmm_rrr_add],
            _need_broadcast_gemm,
        ),
        (
            [permute021, gemm_rrr_bias],
            [ops.gemm_universal.bmm_crr_add, None],
            _need_broadcast_gemm,
        ),
    )

    graph_transformed = True
    while graph_transformed:
        graph_transformed = False
        for source, targets, condition in permute_mm_patterns:
            fused, sorted_graph = _fuse_permute_bmm_ops(
                sorted_graph, source, targets, condition
            )
            graph_transformed |= fused

    return sorted_graph
