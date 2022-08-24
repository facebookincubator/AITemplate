# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Elementwise operator definition, which covers UNARY / Binary / Ternary operators.
"""
from typing import Any, List

from ....utils import shape_utils
from ...base import IntVar, Operator, Tensor
from .epilogue import FuncEnum

# pylint: disable=C0103,W0221,W0102,C0301,W0223,R1724


class elementwise(Operator):
    """elementwise operator definition."""

    def __init__(self, func_enum: FuncEnum) -> None:
        """
        Parameters
        ----------
        func_enum : the underlying function enum.
        """

        super().__init__()
        self._attrs["op"] = "elementwise"
        self._attrs["func"] = func_enum
        self._attrs["has_profiler"] = False

    def _infer_shapes(self, *args: Tensor) -> List[IntVar]:
        """Offline shape inference."

        Parameters
        ----------
        args : input tensors.

        Returns
        -------
        List[IntVar] : output tensor shape.
        """

        if len(args) == 0:
            raise RuntimeError(
                "Elementwise op {} doesn't have inputs!".format(self._attrs["func"])
            )
        max_shape = None
        for tensor in args:
            shape = tensor._attrs["shape"]
            if max_shape is None:
                max_shape = list(shape)
            broadcastable, max_shape = shape_utils.get_broadcast_max_shape(
                max_shape, shape
            )
            if not broadcastable:
                raise RuntimeError(
                    "Tensor shapes of elementwise ops are not compatible! Shape1: {}, shape2: {}".format(
                        max_shape, shape
                    )
                )
        return max_shape

    def __call__(self, *args: Tensor) -> Tensor:
        self._attrs["args"] = list(args)
        self._attrs["inputs"] = [arg for arg in args if not arg.is_a_const_num()]
        self._set_depth()
        output_shape = self._infer_shapes(*args)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def replace_input_tensor(self, old_tensor, new_tensor) -> None:
        super().replace_input_tensor(old_tensor, new_tensor)
        self._attrs["args"] = [
            new_tensor if tensor is old_tensor else tensor
            for tensor in self._attrs["args"]
        ]

    def _args_for_pseudo_code(self):
        return [self._attrs["func"]]


class clamp(Operator):
    """Clamps all elements in input into the range [min_value, max_value].
    Returns y = min(max(x, min_value), max_value).
    If min is None, there is no lower bound. Or, if max is None there is no upper bound.
    If min is greater than max torch.clamp(..., min, max) sets all elements in input to
    the value of max.
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "clamp"
        self._attrs["has_profiler"] = False

    def __call__(
        self, x: Tensor, min_value: Any = None, max_value: Any = None
    ) -> Tensor:
        if isinstance(min_value, (int, float)):
            min_value = Tensor(value=min_value, shape=[])
        if isinstance(max_value, (int, float)):
            max_value = Tensor(value=max_value, shape=[])
        if min_value is None and max_value is not None:
            return elementwise(FuncEnum.MIN)(
                x,
                max_value,
            )
        if max_value is None and min_value is not None:
            return elementwise(FuncEnum.MAX)(
                x,
                min_value,
            )
        assert not (max_value is None and max_value is None)
        return elementwise(FuncEnum.MIN)(
            elementwise(FuncEnum.MAX)(x, min_value),
            max_value,
        )
