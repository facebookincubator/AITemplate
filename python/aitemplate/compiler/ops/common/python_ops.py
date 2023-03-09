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
Syntax sugar ops to support List/Tuples in the IR. These ops don't generate any code.
"""
from typing import Any, List, Tuple, Union

from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor

from aitemplate.utils.tensor_utils import wrap_dim

# pylint: disable=C0103,W0221,R1732,W0613


class getitem(Operator):
    """Retrieve a single element from a list of tuple at a certain index."""

    def __call__(self, vals: Union[List[Any], Tuple[Any]], index: int) -> Any:
        assert isinstance(vals, (tuple, list))
        assert len(vals) > 0

        wrapped_idx = wrap_dim(int(index), len(vals))
        val = vals[wrapped_idx]
        if isinstance(val, Tensor) or isinstance(val, (IntVar, IntImm)):
            return val
        else:
            raise NotImplementedError(
                "getitem op does not support this val type: {}".format(val)
            )


class tuple_construct(Operator):
    """Construct a tuple of tensors."""

    def __call__(self, *args: Union[Tensor, IntVar]) -> Tuple[Tensor]:
        outputs = tuple(args)
        return outputs


class list_construct(Operator):
    """Construct a list of tensors."""

    def __call__(self, *args: Union[Tensor, IntVar]) -> List[Tensor]:
        outputs = list(args)
        return outputs
