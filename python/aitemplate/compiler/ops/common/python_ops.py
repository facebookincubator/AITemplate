# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Syntax sugar ops to support List/Tuples in the IR. These ops don't generate any code.
"""
from typing import Any, List, Tuple, Union

from ....utils.tensor_utils import wrap_dim
from ...base import IntImm, IntVar, Operator, Tensor

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
