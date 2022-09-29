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
Utils for unit tests.
"""
from typing import Any, Dict, List

import torch

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor
from aitemplate.utils.graph_utils import get_sorted_ops


DTYPE_TO_TORCH_DTYPE: Dict[str, torch.dtype] = {
    "float16": torch.half,
    "float": torch.float,
    "int": torch.int,
}


def dtype_to_torch_dtype(dtype):
    if dtype is None:
        return None
    torch_dtype = DTYPE_TO_TORCH_DTYPE.get(dtype)
    if torch_dtype is None:
        raise RuntimeError("Unsupported dtype: {}".format(dtype))
    return torch_dtype


def get_random_torch_tensor(shape, dtype):
    if dtype == "float16":
        return torch.randn(shape).cuda().half()
    if dtype == "float":
        return torch.randn(shape).cuda().float()
    if dtype == "int":
        return torch.randn(shape).cuda().int()
    raise RuntimeError("unsupported dtype: {}".format(dtype))


def has_op(sorted_ops: List[Operator], op_name: str) -> bool:
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type == op_name:
            return True
    return False


def graph_has_op(graph: List[Tensor], op_name: str) -> bool:
    return has_op(get_sorted_ops(graph), op_name)


def count_ops(sorted_ops: List[Operator], op_name: str):
    count = 0
    for op in sorted_ops:
        op_type = op._attrs["op"]
        if op_type == op_name:
            count += 1
    return count


def gen_input_tensor(shape: List[Any], name: str = None) -> Tensor:
    tensor = Tensor(
        shape=shape,
        dtype="float16",
        name=name,
        is_input=True,
    )
    return tensor


def get_src_op(tensor: Tensor) -> str:
    assert len(tensor._attrs["src_ops"]) == 1
    return list(tensor._attrs["src_ops"])[0]


def get_src_op_name(tensor: Tensor) -> str:
    return get_src_op(tensor)._attrs["op"]


def get_src_input(tensor: Tensor) -> str:
    src_op = get_src_op(tensor)
    assert len(src_op._attrs["inputs"]) >= 1
    return src_op._attrs["inputs"][0]


def get_shape(shape: List[IntVar], dim_to_value_dict: Dict[str, int]):
    res = [
        dim.value()
        if isinstance(dim, IntImm)
        else dim_to_value_dict[dim._attrs["name"]]
        for dim in shape
    ]
    return res
