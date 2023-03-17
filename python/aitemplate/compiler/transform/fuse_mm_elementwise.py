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
Fuse GEMM with elementwise operations
"""
from typing import List

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.ops.gemm_universal import gemm_rcr_bias_swish

from aitemplate.compiler.transform.fuse_mm_elementwise_patterns import (
    get_gemm_rcr_bias_patterns,
    get_patterns,
)
from aitemplate.compiler.transform.fuse_utils import (
    extract_only_one_op,
    is_elementwise_type,
    transform_simple_fusion_patterns,
)
from aitemplate.compiler.transform.transform_utils import (
    copy_tensor_attributes,
    remove_dst_op_from_tensor,
    remove_single_tensor_op_from_sorted_graph,
    replace_tensor,
    sanitize_sorted_graph,
)

# pylint: disable=C0103,C0415,W0612


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

        gemm_op = extract_only_one_op(tensor._attrs["src_ops"])
        if gemm_op is None:
            continue
        if gemm_op._attrs["op"] != "gemm_rcr_bias":
            continue

        dst_op = list(tensor._attrs["dst_ops"])
        dst_op_size = len(dst_op)
        if dst_op_size not in [1, 2]:
            continue
        swish_tensor = None
        for idx in range(dst_op_size):
            other_idx = (idx + 1) % 2
            if is_elementwise_type(dst_op[idx], FuncEnum.SIGMOID):
                if not is_elementwise_type(dst_op[other_idx], FuncEnum.MUL):
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
            elif is_elementwise_type(dst_op[idx], FuncEnum.SILU):
                swish_tensor = dst_op[idx]._attrs["outputs"][0]
                break

        if swish_tensor is None:
            continue

        gemm_inputs = gemm_op._attrs["inputs"]
        remove_dst_op_from_tensor(gemm_inputs, gemm_op)
        # Output of sigmoid and final mul of swish.
        for i in range(dst_op_size):
            to_remove.add(dst_op[i]._attrs["outputs"][0])

        new_tensor = gemm_rcr_bias_swish()(*gemm_inputs)
        copy_tensor_attributes(new_tensor, swish_tensor)
        replace_tensor(swish_tensor, new_tensor)
        new_sorted_graph[-1] = new_tensor

    return sanitize_sorted_graph(new_sorted_graph)


def _transform_gemm_bias(sorted_graph: List[Tensor]) -> List[Tensor]:
    return transform_simple_fusion_patterns(sorted_graph, get_gemm_rcr_bias_patterns())


def _transform_mm_elementwise(sorted_graph: List[Tensor]) -> List[Tensor]:
    fusion_patterns = get_patterns()

    return transform_simple_fusion_patterns(sorted_graph, fusion_patterns)


def fuse_mm_elementwise(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """Fuse GEMMs with elementwise operations.

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
        _fuse_bmm_mul_or_div_alpha,
        _transform_gemm_bias,
        _transform_mm_elementwise,
        _fuse_gemm_rcr_bias_swish,
    ]
    for func in funcs:
        sorted_graph = func(sorted_graph)
    return sorted_graph
