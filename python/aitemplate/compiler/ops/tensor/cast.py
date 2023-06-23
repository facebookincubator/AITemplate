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


class cast(Operator):
    """
    Returns the cast of input tensor to specified type.
    Only the conversion between any pair of float16, bfloat16,
    and float32 dtypes is supported.

    Args:
        x (Tensor): the source tensor
        dtype (str): the target type for the cast operator

    Returns:
        Tensor: a tensor with the type converted to the
        specified dtype.

    """

    def __init__(self) -> None:
        super().__init__()

        self._attrs["op"] = "cast"
        self._attrs["has_profiler"] = False

    def __call__(
        self,
        x: Tensor,
        dtype: str = "bfloat16",
    ) -> Tensor:
        x_dtype = normalize_dtype(x._attrs["dtype"])
        dtype = normalize_dtype(dtype)
        if x_dtype not in ("float16", "bfloat16", "float32"):
            raise TypeError(
                f"Expected dtype for x must be float16,bfloat16 or float32 , but got {x_dtype}."
            )

        if dtype not in ("float16", "bfloat16", "float32"):
            raise TypeError(
                f"Expected dtype to cast must be float16,bfloat16 or float32 , but got {dtype}."
            )
        if dtype == x_dtype:
            return x

        self._attrs["inputs"] = [x]
        self._attrs["cast_dtype"] = dtype
        self._set_depth()

        output_shape = x._attrs["shape"]
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=dtype,
        )
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
