# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Perform graph transformation specifically for gemm -> gemm_special.
Check each transform function summary for specific pattern to be transformed.
"""
from typing import Callable, List, Tuple, Type

from .. import ops
from ..base import Operator, Tensor
from ..ops.gemm_special.gemm_rrr_small_nk import gemm_rrr_small_nk
from ..ops.gemm_universal.bmm_rcr import bmm_rcr
from ..ops.gemm_universal.gemm_rrr import gemm_rrr
from .transform_utils import (
    copy_src_op_attributes,
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    replace_tensor,
    sanitize_sorted_graph,
)

# pylint: disable=C0103,C0415,W0612


def _simple_transform_with_constraint(
    sorted_graph: List[Tensor],
    matchFunc: Callable[[Tensor], bool],
    replaceFunc: Callable[[Tensor], Tensor],
) -> List[Tensor]:
    """
    Replace ops in sorted_graph that match constraint provided by matchFunc.
    Op to be replaced is determined by matchFunc, which if true, we call replaceFunc to substitute the tensor away.


    Parameters
    ----------
    sorted_graph : List[Tensor]
        original AIT graph

    matchFunc : Callable(Tensor) -> bool
        A function that returns whether or not the operator needs to be substituted

    replaceFunc  : Callable(Tensor) -> Tensor
        A function that return the resulting tensor that replaces the original one.

    Returns
    ----------
    List[Tensor]
        AIT graph after transformation
    """

    new_sorted_graph = []
    transformed = False
    for tensor in sorted_graph:
        if matchFunc(tensor):
            new_sorted_graph.append(replaceFunc(tensor))
            transformed = True
        else:
            new_sorted_graph.append(tensor)

    if not transformed:
        return sorted_graph
    return sanitize_sorted_graph(new_sorted_graph)


def _single_source_shape_constraint_funcs(
    src_type: Type[Operator], target_type: Type[Operator]
) -> Tuple[Callable[[Tensor], bool], Callable[[Tensor], Tensor]]:
    """
    Returns matching function and replace function for tensors that are single src_op with shape constraints.

    Parameters
    ----------
    src_type : Type[Operator]
        An operator type that can be substituted by target_type provided tensor shape constraint is satisfied.
    target_type : Type[Operator]
        An operator that can substitute src_type provided shape constraint is satisfied.
        is_valid_shape needs to be implemented for target_type

    Returns
    ----------
    Tuple[Callable[[Tensor], bool], Callable[[Tensor], Tensor]]
        A tuple of function which corresponds to a matching function and a replacing function wrt tensor provided.
    """

    def matchFunc(tensor: Tensor) -> bool:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            return False

        src_op = list(src_ops)[0]
        if src_op._attrs["op"] != src_type()._attrs["op"]:
            return False

        A, B = src_op._attrs["inputs"]
        if not target_type.is_valid_shape(A, B):
            return False

        return True

    def replaceFunc(old_tensor: Tensor) -> Tensor:
        src_op = list(old_tensor._attrs["src_ops"])[0]
        A, B = src_op._attrs["inputs"]

        new_op = target_type()
        new_tensor = new_op(A, B)
        copy_tensor_attributes(new_tensor, old_tensor)
        copy_src_op_attributes(new_tensor, old_tensor)
        remove_dst_op_from_tensor([A, B], src_op)
        replace_tensor(old_tensor, new_tensor)
        return new_tensor

    return (matchFunc, replaceFunc)


def _transform_bmm_rcr_n1(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Replace kernel bmm_rcr with N == 1 and K % 8 == 0 with bmm_rcr_n1

    Parameters
    ----------
    sorted_graph : List[Tensor]
        original AIT graph

    Returns
    ----------
    List[Tensor]
        AIT graph with suitable bmm_rcr substituted with bmm_rcr_n1
    """
    matchFunc, replaceFunc = _single_source_shape_constraint_funcs(
        bmm_rcr, ops.gemm_special.bmm_rcr_n1
    )

    return _simple_transform_with_constraint(sorted_graph, matchFunc, replaceFunc)


def _transform_gemm_rrr_small_nk(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    Replace kernel gemm_rrr with N <= 8, K <= 16 with gemm_rrr_small_nk

    Parameters
    ----------
    sorted_graph : List[Tensor]
        original AIT graph

    Returns
    ----------
    List[Tensor]
        AIT graph with suitable gemm_rrr substituted with gemm_rrr_small_nk
    """
    matchFunc, replaceFunc = _single_source_shape_constraint_funcs(
        gemm_rrr, gemm_rrr_small_nk
    )

    return _simple_transform_with_constraint(sorted_graph, matchFunc, replaceFunc)


def transform_special_ops(sorted_graph: List[Tensor]) -> List[Tensor]:
    funcs = [
        _transform_bmm_rcr_n1,
        _transform_gemm_rrr_small_nk,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
