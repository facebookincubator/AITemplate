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

from enum import Enum

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.dtype import normalize_dtype


class RelationalEnum(Enum):
    GE = ">="
    LE = "<="
    LT = "<"
    GT = ">"
    EQ = "=="
    NE = "!="


class relational(Operator):
    """
    Relational operator that supports comparing a tensor to another tensor or a constant

    Parameters:
        left (Tensor): the tensor to compare

        right (Tensor or float): the tensor or value to compare

    Returns:
        Tensor: a tensor of bool
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "relational"

    def __call__(self, left: Tensor, right: Tensor) -> Tensor:
        assert self._attrs["func"] is not None, "No function registered"
        common_dtype = None
        assert isinstance(
            left, Tensor
        ), "Relational expects left operand to be a Tensor"
        common_dtype = normalize_dtype(left.dtype())
        left._attrs["dtype"] = common_dtype

        if isinstance(right, int) or isinstance(right, float):
            right = Tensor(shape=[], value=right, dtype=common_dtype)
        else:
            assert isinstance(
                right, Tensor
            ), "Relational expects right operand to be a Tensor or constant"
            assert (
                normalize_dtype(right.dtype()) == common_dtype
            ), f"Type promotions are not supported; got dtype {left.dtype()}, but expected {common_dtype}"
            assert (
                left.shape() == right.shape()
            ), "Relational does not support broadcasting yet. It expects tensor of same shape."
            right._attrs["dtype"] = common_dtype

        self._attrs["args"] = [left, right]
        self._attrs["inputs"] = [left] if right.is_a_const_num() else [left, right]
        self._set_depth()
        output = Tensor(left.shape(), src_ops=[self], dtype="bool")
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)


class ge(relational):
    def __init__(self) -> None:
        super().__init__()
        self._attrs["func"] = RelationalEnum.GE


class le(relational):
    def __init__(self) -> None:
        super().__init__()
        self._attrs["func"] = RelationalEnum.LE


class gt(relational):
    def __init__(self) -> None:
        super().__init__()
        self._attrs["func"] = RelationalEnum.GT


class lt(relational):
    def __init__(self) -> None:
        super().__init__()
        self._attrs["func"] = RelationalEnum.LT


class eq(relational):
    def __init__(self) -> None:
        super().__init__()
        self._attrs["func"] = RelationalEnum.EQ


class ne(relational):
    def __init__(self) -> None:
        super().__init__()
        self._attrs["func"] = RelationalEnum.NE
