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
Eliminate elementwise no-ops (*/1, +-0)
"""
from typing import Callable, Dict, List

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.public import FuncEnum
from aitemplate.compiler.transform import transform_utils


def _is_const_num(tensor: Tensor, val: int) -> bool:
    return tensor.is_a_const_num() and tensor._attrs["value"] == val


def func_add_predicate(src_op: Tensor) -> bool:
    if _is_const_num(src_op._attrs["args"][0], 0) or _is_const_num(
        src_op._attrs["args"][1], 0
    ):
        return True
    return False


def func_sub_predicate(src_op: Tensor) -> bool:
    if _is_const_num(src_op._attrs["args"][1], 0):
        return True
    return False


def func_mul_predicate(src_op: Tensor) -> bool:
    if _is_const_num(src_op._attrs["args"][0], 1) or _is_const_num(
        src_op._attrs["args"][1], 1
    ):
        return True
    return False


def func_div_predicate(src_op: Tensor) -> bool:
    if _is_const_num(src_op._attrs["args"][1], 1):
        return True
    return False


FUNC_TO_PREDICATE_MAP: Dict[FuncEnum, Callable[[Tensor], bool]] = {
    FuncEnum.ADD: func_add_predicate,
    FuncEnum.SUB: func_sub_predicate,
    FuncEnum.MUL: func_mul_predicate,
    FuncEnum.DIV: func_div_predicate,
}


def remove_elementwise_no_ops(
    sorted_graph: List[Tensor], workdir: str = None
) -> List[Tensor]:
    """elementwise no-ops (*/1, +-0)"""
    for tensor in sorted_graph:

        src_ops = tensor._attrs["src_ops"]
        if len(src_ops) != 1:
            continue
        src_op = list(src_ops)[0]

        if (
            src_op._attrs["op"] != "elementwise"
            or src_op._attrs["func"] not in FUNC_TO_PREDICATE_MAP
            or len(src_op._attrs["args"]) != 2  # Skip legacy usecase
        ):
            continue

        predicate = FUNC_TO_PREDICATE_MAP[src_op._attrs["func"]]
        if not predicate(src_op):
            continue

        input_tensor = src_op._attrs["inputs"][0]
        # skip a very special case where ops takes an input and produces an output
        if tensor._attrs["is_output"] and input_tensor._attrs["is_input"]:
            continue
        transform_utils.remove_single_tensor_op_from_sorted_graph(src_op)

    return transform_utils.sanitize_sorted_graph(sorted_graph)
