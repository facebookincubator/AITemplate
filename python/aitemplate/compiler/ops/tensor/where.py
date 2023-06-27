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


from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.compiler.dtype import normalize_dtype


class where(Operator):
    """
    Return a tensor of elements selected from either input or other, depending on condition.

    Parameters:
        condition (A bool Tensor): When True (nonzero), yield input, otherwise yield other

        input_tensor (Tensor or Scalar): value (if input is a scalar) or values selected at indices where condition is True

        other_tensor (Tensor or Scalar): value (if other is a scalar) or values selected at indices where condition is False

        dtype: output dtype if both input_tensor and output_tensor is scalar
    Returns:
        Tensor: A tensor of shape equal to the shape of condition
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "where"

    def __call__(
        self,
        condition: Tensor,
        input_tensor: Tensor,
        other_tensor: Tensor,
        dtype: str = "",
    ) -> Tensor:
        assert isinstance(
            condition, Tensor
        ), f"condition needs to be a tensor, but got {type(condition)}"
        assert (
            condition.dtype() == "bool"
        ), f"condition needs to be a bool tensor, but got {condition.dtype()}"

        output_shape = condition.shape()
        args = []
        inputs = []
        common_dtype = None
        for tensor in [input_tensor, other_tensor]:
            if isinstance(tensor, int) or isinstance(tensor, float):
                tensor = Tensor(shape=[], value=tensor, dtype=common_dtype)
            else:
                assert isinstance(
                    tensor, Tensor
                ), f"Unsupported data type: {type(tensor)}"
                assert (
                    tensor.shape() == output_shape
                ), f"Tensor shape should be the same, {tensor.shape()} != {output_shape}"
                if common_dtype is None:
                    common_dtype = normalize_dtype(tensor.dtype())
                else:
                    assert common_dtype == normalize_dtype(
                        tensor.dtype()
                    ), f"Expect tensor of the same dtype, got {common_dtype} and {normalize_dtype(tensor.dtype())}"
                inputs.append(tensor)

            args.append(tensor)

        # In case where both inputs are scalars,
        if len(inputs) == 0:
            assert dtype != "", "dtype needs to be provided for scalars"
            common_dtype = normalize_dtype(dtype)
            for arg in args:
                arg._attrs["dtype"] = common_dtype
        self._attrs["args"] = [condition, *args]
        self._attrs["inputs"] = [condition, *inputs]
        self._set_depth()
        output = Tensor(
            shape=output_shape,
            src_ops={self},
            dtype=common_dtype,
        )
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
