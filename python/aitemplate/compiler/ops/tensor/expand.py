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
from typing import List, Union

from aitemplate.backend import registry

from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor, Operator, Tensor
from aitemplate.utils.shape_utils import convert_shape_to_IntVar, gen_int_var


def _normalize_dim(dim: IntVar) -> IntVar:
    """
    Convert IntVars with the same upper and lower bounds to IntImms.
    """
    if isinstance(dim, IntImm) or dim.upper_bound() != dim.lower_bound():
        return dim
    return IntImm(dim.upper_bound())


def _dim_has_value(dim: IntVar, value: int) -> bool:
    return isinstance(dim, IntImm) and dim.value() == value


class expand(Operator):
    """
    Expands a tensor's singleton dimensions.

    Expanded dimensions in the input tensor must be `IntImm`s with value() == 1,
    or `IntVar`s with upper_bound() == lower_bound() == 1.
    The output shape may be dynamic.

    The other dimensions in the input must match the input shape exactly,
    or be set to -1.

    Args:
        input (Tensor) : the source tensor
        dim (List[Union[IntImm, IntVar, int]]) : the target dim

    Returns:
        Tensor : the destination tensor

    Example:

    .. highlight:: python
    .. code-block:: python

        x = Tensor([2, 3], name="input_0", is_input=True)
        y = Tensor([2, 3], name="input_1", is_input=True)
        x_expand = ops.expand()(x, [IntImm(1), -1, -1])
        y_expand = ops.expand()(y, [IntVar([1, 1]), -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(x_expand, y_expand)

    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "expand"
        self._attrs["expand_dim"] = None

    @staticmethod
    def _should_reuse_input_dim(dim_tensor: IntVar, dim_arg: IntVar) -> bool:
        return _dim_has_value(dim_arg, -1) or dim_tensor == dim_arg

    def _infer_shape(self, tensor: Tensor, shape: List[IntVar]) -> List[IntVar]:
        output_shape = []
        input_shape = tensor._attrs["shape"]

        if len(shape) != len(input_shape):
            raise ValueError(
                f"Input shape ndim ({len(shape)}) must match tensor's ndim ({len(input_shape)})"
            )

        for i, dim_tensor in enumerate(input_shape):
            dim_arg = shape[i]

            # Convert IntVars with the same upper and lower bounds to IntImm's.
            # This lets us tell that expanding IntImm(1) into IntVar([1, 1]) is
            # actually a no-op.
            dim_tensor = _normalize_dim(dim_tensor)
            dim_arg = _normalize_dim(dim_arg)

            if self._should_reuse_input_dim(dim_tensor, dim_arg):
                output_shape.append(
                    gen_int_var(
                        dim_tensor._attrs["values"], name=dim_tensor._attrs["name"]
                    )
                )
            elif _dim_has_value(dim_tensor, 1):
                if self._attrs["expand_dim"] is not None:
                    raise NotImplementedError(
                        f"Expand only supports expanding one dim. Tried to expand dim {i}, but already expanded dim {self._attrs['expand_dim']}."
                    )
                self._attrs["expand_dim"] = i
                output_shape.append(
                    gen_int_var(dim_arg._attrs["values"], name=dim_arg._attrs["name"])
                )
            else:
                raise ValueError(
                    f"Tried to expand non-singleton dimension {i}. Input tensor dim: {dim_tensor}, target shape dim: {dim_arg}"
                )

        return output_shape

    def __call__(
        self, tensor: Tensor, shape: List[Union[int, IntVar, IntVarTensor]]
    ) -> Tensor:
        self._attrs["inputs"] = [tensor]
        for dim in shape:
            if isinstance(dim, IntVarTensor):
                self._attrs["inputs"].append(dim)
        shape = convert_shape_to_IntVar(shape)
        self._set_depth()
        output_shape = self._infer_shape(tensor, shape)
        output = Tensor(output_shape, src_ops={self}, dtype=tensor._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
