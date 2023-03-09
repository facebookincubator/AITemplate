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
Operator definition for gather.
"""
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor

# pylint: disable=C0103,W0221,W0102,W0223


class gather(Operator):
    """gather implementation

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "gather"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor, dim: int, index: Tensor) -> Tensor:
        dtype = index._attrs["dtype"]
        if dtype != "int64":
            raise RuntimeError(
                "expected dtype int64 for index but got {}".format(dtype)
            )

        x_shape = x._attrs["shape"]
        if dim >= len(x_shape):
            raise RuntimeError(
                "dimension value {} expected to be less than {}".format(
                    dim, len(x_shape)
                )
            )
        self._attrs["inputs"] = [x, index]
        self._attrs["gather_dim"] = dim
        self._set_depth()

        output_shape = index._attrs["shape"]
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=x._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        return output

    def _get_func(self, fmt_str):
        """
        Parameters
        ----------
        inputs : string
            format string to create func_key for looking up func
            from the registry
        """

        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)
