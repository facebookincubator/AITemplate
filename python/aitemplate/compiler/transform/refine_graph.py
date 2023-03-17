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
Graph pass to dedup operators with same signatures.
"""
import logging
from typing import List

from aitemplate.compiler.base import Operator, Tensor

from aitemplate.utils.graph_utils import get_sorted_ops

# pylint: disable=C0103


_LOGGER = logging.getLogger(__name__)

SPECIAL_CHECK_FUNC_KEYS = {
    "inputs",
    "name",
    "depth",
    "outputs",
    "original_inputs",
    "original_outputs",
    "gemm_operand_groups",
    "original_name",
    "f_ab_alignment",
    "elementwise_ops",
    "args",
}


def same_tensor_type(t1: Tensor, t2: Tensor):
    if t1.dtype() != t2.dtype():
        return False
    if t1._attrs["value"] != t2._attrs["value"]:
        return False
    t1s = t1.shape()
    t2s = t2.shape()
    if len(t1s) != len(t2s):
        return False
    for d1, d2 in zip(t1s, t2s):
        if d1 != d2:
            return False
    return True


def check_inputs_outputs(key: str, o1: Operator, o2: Operator):
    # check inputs
    o1_args = o1._attrs[key]
    o2_args = o2._attrs[key]
    if len(o1_args) != len(o2_args):
        return False
    for t1, t2 in zip(o1_args, o2_args):
        if not same_tensor_type(t1, t2):
            return False
    return True


def check_fused_elementwise_ops(o1: Operator, o2: Operator):
    ops1 = o1._attrs["elementwise_ops"]
    ops2 = o2._attrs["elementwise_ops"]

    # Allow single input to simply it
    if len(o1._attrs["inputs"]) != 1:
        return False

    # Disallow multiple ops
    if len(ops1) != len(ops2) or len(ops1) != 1:
        return False

    return same_function_type(ops1[0], ops2[0])


def same_function_type(o1: Operator, o2: Operator):
    if o1._attrs["op"] != o2._attrs["op"]:
        return False

    if len(o1._attrs) != len(o2._attrs):
        return False
    keys = o1._attrs.keys()

    # ban group gemm ops
    if "unique_workspace" in keys:
        return False

    # check general attrs
    for key in keys:
        if key not in o2._attrs:
            return False
        if key not in SPECIAL_CHECK_FUNC_KEYS:
            if o1._attrs[key] != o2._attrs[key]:
                return False

    if not check_inputs_outputs("inputs", o1, o2):
        return False

    # for fused_elementwise ops
    if o1._attrs["op"] == "fused_elementwise" and (
        not check_fused_elementwise_ops(o1, o2)
    ):
        return False
    if "original_inputs" in keys:
        if not check_inputs_outputs("original_inputs", o1, o2):
            return False
    if "original_outputs" in keys:
        if not check_inputs_outputs("original_outputs", o1, o2):
            return False

    # for elementwise ops
    if "args" in keys:
        if not check_inputs_outputs("args", o1, o2):
            return False

    return True


def refine_graph(sorted_graph: List[Tensor]):
    """Graph pass to dedup operators with same signatures.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph
    """
    sorted_ops = get_sorted_ops(sorted_graph)

    exist_func = []
    refined_ops = 0
    total_ops = len(sorted_ops)

    refined_ops_set = set()

    for func in sorted_ops:
        found = False
        for f in reversed(exist_func):
            if same_function_type(f, func):
                func._attrs["name"] = f._attrs["name"]
                found = True
                refined_ops += 1
                break
        if not found:
            exist_func.append(func)
        if found:
            refined_ops_set.add(func._attrs["op"])

    _LOGGER.debug(f"refined ops: {refined_ops_set}")
    _LOGGER.info(f"reduced unique ops from {total_ops} to {total_ops - refined_ops}")
