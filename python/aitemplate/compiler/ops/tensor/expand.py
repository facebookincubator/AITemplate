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
from enum import IntEnum
from typing import List, Union

from aitemplate.backend import registry

from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor, Operator, Tensor
from aitemplate.utils.shape_utils import convert_shape_to_IntVar


def _normalize_dim(dim: IntVar) -> IntVar:
    """
    Convert IntVars with the same upper and lower bounds to IntImms.
    """
    if isinstance(dim, IntImm) or dim.upper_bound() != dim.lower_bound():
        return dim
    return IntImm(dim.upper_bound())


def _dim_has_value(dim: IntVar, value: int) -> bool:
    return isinstance(dim, IntImm) and dim.value() == value


class ExpandDimensionType(IntEnum):
    ADD_DIM = 0
    EXPAND_DIM = 1
    KEEP_DIM = 2


class expand(Operator):
    """
    Expands a tensor's singleton dimensions.

    Expanded dimensions in the input tensor must be `IntImm`s with value() == 1,
    or `IntVar`s with upper_bound() == lower_bound() == 1.
    The output shape may be dynamic.

    The other dimensions in the input must match the input shape exactly,
    or be set to -1, in which case the output shape is unchanged for that dimension.

    Tensor can be also expanded to a larger number of dimensions, and the new ones will
    be appended at the front. For the new dimensions, the size cannot be set to -1.

    Args:
        input (Tensor) : the source tensor
        shape (List[Union[IntImm, IntVar, int]]) : target shape ( dimensions with size -1 will be kept, excess dimensions are added at the front )
        index_type (str): Native type used for indices, may be "int64" (default) or "int32".
                          Pick "int32" only if the total number of elements is lower than 2^31
        optimize_fixed_dims (bool) : if True, and if the conditions are given, allow to apply optimizatins assuming mostly fixed shapes.
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

    def _infer_shape(self, tensor: Tensor, target_shape: List[IntVar]) -> List[IntVar]:
        output_shape = []
        input_shape = tensor._attrs["shape"]
        assert len(input_shape) > 0, "Input tensor must have a shape of length > 0"
        for i, dim in enumerate(input_shape):
            if dim.lower_bound() < 0:
                raise ValueError(
                    f"Dimension {i} of expand input tensor shape has range [{dim.lower_bound()}:{dim.upper_bound()}], which includes negative values."
                )
        for i, dim in enumerate(target_shape):
            if dim.lower_bound() < 0 and not (
                dim.lower_bound() == -1 and dim.upper_bound() == -1
            ):
                raise ValueError(
                    f"Dimension {i} of expand target shape has range [{dim.lower_bound()}:{dim.upper_bound()}], which includes negative values."
                )

        if len(target_shape) < len(input_shape):
            raise ValueError(
                f"Target shape length ({len(target_shape)}) must be greater or equal to input tensor's shape length ({len(input_shape)})"
            )
        add_ndims = len(target_shape) - len(input_shape)
        for i, dim_to_add in enumerate(target_shape[:add_ndims]):
            if dim_to_add.lower_bound() <= 0:
                raise ValueError(
                    f"Output shape dimension {i} to be added has value range [{dim_to_add.lower_bound()}:{dim_to_add.upper_bound()}], but violates constraint that it must be greater or equal to 1."
                )
            output_shape.append(dim_to_add)
        self._attrs["dim_types"] = [
            ExpandDimensionType.ADD_DIM
        ] * add_ndims  # 0 meaning, dimension is added
        for i, dim_input in enumerate(input_shape):
            dim_target = target_shape[i + add_ndims]

            # Convert IntVars with the same upper and lower bounds to IntImm's.
            # This lets us tell that expanding IntImm(1) into IntVar([1, 1]) is
            # actually a no-op.
            dim_input = _normalize_dim(dim_input)
            dim_target = _normalize_dim(dim_target)

            if self._should_reuse_input_dim(dim_input, dim_target):
                output_shape.append(
                    dim_input
                )  # no deepcopy, dim symbol should be identical
                self._attrs["dim_types"].append(
                    ExpandDimensionType.KEEP_DIM
                )  # 2 meaning, dimension is kept as is
            elif _dim_has_value(dim_input, 1):
                output_shape.append(dim_target)
                self._attrs["dim_types"].append(
                    ExpandDimensionType.EXPAND_DIM
                )  # 1 meaning, dimension is expanded
            else:
                raise ValueError(
                    f"Tried to expand non-singleton dimension {i}. Input tensor dim: {dim_input}, target shape dim: {dim_target}"
                )
        head_dim_count = 0
        head_size = 1
        for dim_type, dim in zip(self._attrs["dim_types"], output_shape):
            if dim_type == ExpandDimensionType.KEEP_DIM and dim.lower_bound() != 1:
                break
            head_size *= dim.lower_bound()
            head_dim_count += 1
        self._attrs["head_dim_count"] = head_dim_count
        self._attrs["head_size"] = head_size
        self._attrs["non_head_dims_are_fixed"] = all(
            dim.lower_bound() == dim.upper_bound() for dim in output_shape[add_ndims:]
        )
        return output_shape

    def __call__(
        self,
        tensor: Tensor,
        shape: List[Union[int, IntVar, IntVarTensor]],
        index_type="int64",
        optimize_fixed_dims=True,
    ) -> Tensor:
        self._attrs["inputs"] = [tensor]
        self._attrs["index_type"] = index_type
        self._attrs["optimize_fixed_dims"] = optimize_fixed_dims
        for dim in shape:
            if isinstance(dim, IntVarTensor):
                self._attrs["inputs"].append(dim)
        shape = convert_shape_to_IntVar(shape)
        if index_type not in ["int64", "int32"]:
            raise ValueError("index_type for expand op has to be int64_t or int32_t")
        self._set_depth()

        output_shape = self._infer_shape(tensor, shape)
        output = Tensor(output_shape, src_ops={self}, dtype=tensor._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
