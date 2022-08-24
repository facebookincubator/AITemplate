# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any, Dict, List

from aitemplate.compiler.base import IntImm, IntVar, Tensor


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
