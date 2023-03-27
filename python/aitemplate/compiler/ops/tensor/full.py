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

from typing import List

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntVar, Operator, Tensor
from aitemplate.compiler.dtype import get_dtype_size


class full(Operator):
    """
    Creates a tensor of a given `shape` and `dtype` filled
    with the specified `fill_value` (float scalar).

    Args:
        shape (int or IntVar or List[IntVar]): the shape of the output Tensor.
        fill_value (int or float): the value to fill the output Tensor with.
        dtype (str): the dtype of the output Tensor.

    Returns:
        Tensor: a tensor of `shape` and `dtype` filled with `fill_value`.
    """

    def __init__(self) -> None:
        super().__init__()

        self._attrs["op"] = "full"
        self._attrs["has_profiler"] = False

    def __call__(
        self,
        shape: List[IntVar],
        fill_value: float,
        dtype: str = "float16",
    ) -> Tensor:
        if isinstance(shape, (int, IntVar)):
            shape = [shape]
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape must be List[IntVar], but got {shape}.")
        shape = list(shape)

        if not isinstance(fill_value, (int, float)):
            raise TypeError(f"fill_value must be a scalar, but got {fill_value}.")
        fill_value = float(fill_value)

        # validation inside
        get_dtype_size(dtype)

        self._attrs["inputs"] = []
        self._attrs["fill_value"] = fill_value

        self._set_depth()
        output = Tensor(shape, src_ops={self}, dtype=dtype)
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
