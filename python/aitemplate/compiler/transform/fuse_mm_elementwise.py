# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from typing import Any, List, Set, Union

from ..base import Operator, Tensor
from ..ops.common import elementwise
from ..ops.common.epilogue import FuncEnum
from ..ops.gemm_universal import gemm_rcr, gemm_rcr_bias, gemm_rcr_bias_swish

from .fuse_mm_elementwise_patterns import get_patterns
from .toposort import toposort
from .transform_utils import (
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    remove_single_tensor_op_from_sorted_graph,
    replace_tensor,
    sanitize_sorted_graph,
)

# pylint: disable=C0103,C0415,W0612


def _extract_only_one_op(ops: Set[Union[None, Operator]]) -> Union[None, Operator]:
    """
    Helper function that returns the op from src_ops() or dst_ops() call.
    Return None if there are no ops or if there's more than one op.
    """
    if ops is None or len(ops) != 1:
        return None
    return list(ops)[0]


def _fuse_bmm_mul_or_div_alpha(sorted_graph: List[Tensor]) -> List[Tensor]:
    """This pass fuses bmm and mul (or div) if mul's other operand is a
       constant scalar tensor (i.e. which has a valid "value" attribute.
       In such a case, we turn this constant value into bmm's alpha.
       Note that for div cases, we assign 1/const_val to alpha.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        input sorted graph

    Return
    ----------
    List[Tensor]
        modified sorted graph upon success. Otherwise, the original sorted
        graph will be returned.
    """
    for tensor in sorted_graph:
        src_ops = tensor._attrs["src_ops"]
        if src_ops is None or len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]
        if src_op is None:
            continue
        if not src_op._attrs["op"].startswith("bmm"):
            continue
        bmm_op = src_op

        dst_ops = list(tensor._attrs["dst_ops"])
        if not dst_ops or len(dst_ops) != 1:
            continue

        next_op = dst_ops[0]
        if next_op._attrs["op"] != "elementwise":
            continue
        if next_op._attrs["func"] == FuncEnum.MUL:
            is_div = False
        elif next_op._attrs["func"] == FuncEnum.DIV:
            is_div = True
        else:
            continue

        elem_op = next_op
        elem_inputs = elem_op._attrs["inputs"]
        if len(elem_inputs) != 1:
            continue
        elem_args = elem_op._attrs["args"]
        if len(elem_args) != 2:
            continue
        # make sure cst_tensor is the divisor of the DIV op
        if is_div and tensor == elem_args[1]:
            continue
        cst_tensor = elem_args[1] if tensor == elem_args[0] else elem_args[1]
        # skip non-constant scalar tensor
        if not cst_tensor.is_a_const_num():
            continue
        cst_val = cst_tensor._attrs["value"]
        # let's only consider int and float builtin types. Seems that it doesn't
        # make any sense to take other scalar types like str and convert it
        # to a float.
        if not isinstance(cst_val, (float, int)):
            continue
        # OK, we are good so let's add cst_val to bmm's alpha attribute
        bmm_op._attrs["alpha"] = 1.0 / float(cst_val) if is_div else float(cst_val)
        # remove this MUL/DIV
        remove_single_tensor_op_from_sorted_graph(elem_op)

    return sanitize_sorted_graph(sorted_graph)


def _is_elementwise_type(op: Operator, elementwise_type):
    if op._attrs["op"] != "elementwise":
        return False
    return op._attrs["func"] == elementwise_type


def _is_same_op_type(op_A: Operator, op_B: Operator):
    """
    Compare whether 2 ops are of same type.
    """
    if op_A._attrs["op"] != op_B._attrs["op"]:
        return False
    if op_A._attrs["op"] == "elementwise":
        if op_A._attrs["func"] != op_B._attrs["func"]:
            return False

    return True


def _find_fusion_root(tensor: Tensor, fusion_patterns: List[Any]) -> int:
    fusion_idx = -1

    src_op = _extract_only_one_op(tensor._attrs["src_ops"])
    if src_op is None:
        return fusion_idx

    for idx, fusion_pattern in enumerate(fusion_patterns):
        pattern, _ = fusion_pattern
        curr_op = src_op
        curr_tensor = tensor

        for step, pattern_op in enumerate(pattern):
            if not _is_same_op_type(curr_op, pattern_op):
                break
            check_input = getattr(pattern_op, "is_valid_inputs", None)
            if check_input is not None:
                valid, _ = check_input(*curr_op._attrs["inputs"])
                if not valid:
                    break

            if step == len(pattern) - 1:
                fusion_idx = idx
                break

            dst_op = _extract_only_one_op(curr_tensor._attrs["dst_ops"])
            if dst_op is None:
                break
            curr_op = dst_op
            dst_op_tensor = dst_op._attrs["outputs"]
            if len(dst_op_tensor) != 1:
                break
            curr_tensor = dst_op_tensor[0]

        if fusion_idx != -1:
            return fusion_idx

    return fusion_idx


def _transform_simple_fusion_patterns(
    sorted_graph: List[Tensor], fusion_patterns: List[Any]
) -> List[Tensor]:
    output_tensors = []
    to_remove = set()
    for tensor in sorted_graph:
        if tensor in to_remove:
            to_remove.remove(tensor)
            continue

        if tensor._attrs["is_output"]:
            output_tensors.append(tensor)
            continue

        fusion_idx = _find_fusion_root(tensor, fusion_patterns)
        if fusion_idx == -1:
            continue

        to_remove_candidate = set()
        to_remove_dst_op = {}

        mm_op = _extract_only_one_op(tensor._attrs["src_ops"])
        inputs = list(mm_op._attrs["inputs"])
        to_remove_dst_op[mm_op] = list(inputs)

        last_tensor = tensor
        to_remove_candidate.add(last_tensor)

        for _ in range(len(fusion_patterns[fusion_idx][0]) - 1):
            # The check is done in _find_fusion_root, therefore we only need to
            # know how many steps to go forward.
            next_op = _extract_only_one_op(last_tensor._attrs["dst_ops"])
            if next_op._attrs["op"] == "elementwise":
                next_op_inputs = next_op._attrs["args"]
            else:
                next_op_inputs = next_op._attrs["inputs"]
            assert (
                len(next_op_inputs) <= 2 and len(next_op_inputs) > 0
            ), "next_op in pattern should have input length of 1 or 2, got {} instead".format(
                len(next_op_inputs)
            )
            if len(next_op_inputs) == 2:
                # This is the case of add/mul/etc. we put them into inputs.
                if next_op_inputs[0] is last_tensor:
                    other_tensor = next_op_inputs[1]
                elif next_op_inputs[1] is last_tensor:
                    other_tensor = next_op_inputs[0]
                else:
                    raise AssertionError("input does not come from upstream node")
                inputs.append(other_tensor)

                if next_op in to_remove_dst_op:
                    to_remove_dst_op[next_op].append(other_tensor)
                else:
                    to_remove_dst_op[next_op] = [other_tensor]

            last_tensor = next_op._attrs["outputs"][0]
            to_remove_candidate.add(last_tensor)

        # A final check to make sure our replacement is valid.
        new_op = fusion_patterns[fusion_idx][1]

        check_inputs_func = getattr(new_op, "is_valid_inputs", None)
        if check_inputs_func is not None:
            valid, _ = check_inputs_func(*inputs)
            if not valid:
                continue

        # inputs here might not be ready in graph. But we will toposort again
        # at end of pass so it's okay.
        new_tensor = new_op()(*inputs)
        copy_tensor_attributes(new_tensor, last_tensor)
        if new_tensor._attrs["is_output"]:
            output_tensors.append(new_tensor)
        replace_tensor(last_tensor, new_tensor)
        for dst_op, tensors in to_remove_dst_op.items():
            remove_dst_op_from_tensor(tensors, dst_op)
        to_remove |= to_remove_candidate

    new_sorted_graph = toposort(output_tensors)
    return sanitize_sorted_graph(new_sorted_graph)


def _fuse_gemm_rcr_bias_swish(sorted_graph: List[Tensor]) -> List[Tensor]:
    """
    gemm_rcr_bias_swish(A, B) is equivalent to:
        x = gemm_rcr_bias(A, B)
        x1 = sigmoid(x)
        return elementwise(MUL)(x, x1)
    """
    new_sorted_graph = []

    to_remove = set()
    for tensor in sorted_graph:
        if tensor in to_remove:
            continue
        new_sorted_graph.append(tensor)

        if tensor._attrs["is_output"]:
            continue

        gemm_op = _extract_only_one_op(tensor._attrs["src_ops"])
        if gemm_op is None:
            continue
        if gemm_op._attrs["op"] != "gemm_rcr_bias":
            continue

        dst_op = list(tensor._attrs["dst_ops"])
        if len(dst_op) != 2:
            continue
        swish_tensor = None
        for idx in range(2):
            other_idx = (idx + 1) % 2
            if _is_elementwise_type(dst_op[idx], FuncEnum.SIGMOID):
                if not _is_elementwise_type(dst_op[other_idx], FuncEnum.MUL):
                    continue

                is_swish = False
                output = dst_op[idx]._attrs["outputs"][0]
                mul_inputs = dst_op[other_idx]._attrs["inputs"]
                if mul_inputs[0] == output and mul_inputs[1] == tensor:
                    is_swish = True
                if mul_inputs[1] == output and mul_inputs[0] == tensor:
                    is_swish = True
                if not is_swish:
                    continue

                swish_tensor = dst_op[other_idx]._attrs["outputs"][0]
                break

        if swish_tensor is None:
            continue

        gemm_inputs = gemm_op._attrs["inputs"]
        remove_dst_op_from_tensor(gemm_inputs, gemm_op)
        # Output of sigmoid and final mul of swish.
        to_remove.add(dst_op[0]._attrs["outputs"][0])
        to_remove.add(dst_op[1]._attrs["outputs"][0])

        new_tensor = gemm_rcr_bias_swish()(*gemm_inputs)
        copy_tensor_attributes(new_tensor, swish_tensor)
        replace_tensor(swish_tensor, new_tensor)
        new_sorted_graph[-1] = new_tensor

    return sanitize_sorted_graph(new_sorted_graph)


def _transform_gemm_bias(sorted_graph: List[Tensor]) -> List[Tensor]:
    gemm_rcr_bias_patterns = [
        (
            (gemm_rcr(), elementwise(FuncEnum.ADD)),
            gemm_rcr_bias,
        ),
    ]

    return _transform_simple_fusion_patterns(sorted_graph, gemm_rcr_bias_patterns)


def _transform_mm_elementwise(sorted_graph: List[Tensor]) -> List[Tensor]:
    fusion_patterns = get_patterns()

    return _transform_simple_fusion_patterns(sorted_graph, fusion_patterns)


def fuse_mm_elementwise(sorted_graph: List[Tensor]) -> List[Tensor]:
    funcs = [
        _fuse_bmm_mul_or_div_alpha,
        _transform_gemm_bias,
        _transform_mm_elementwise,
        _fuse_gemm_rcr_bias_swish,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
