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
Elementwise operator definition, which covers UNARY / Binary / Ternary operators.
"""
import functools
from typing import Any, List

from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor, Operator, Tensor
from aitemplate.compiler.dtype import normalize_dtype
from aitemplate.compiler.op_registry import OP_REGISTRY
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.ops.common.int_elementwise import INT_ELEMENTWISE_FUNC

from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,C0301,W0223,R1724


def _discover_implicit_jagged_inputs(inputs: List[Tensor]):
    """
    Convert implicit jagged Tensor inputs into explicit jagged Tensors.

    There may be cases when elementwise has both explicit jagged Tensor
    inputs (i.e. with a JaggedIntVar as the first dimension in the shape)
    and "implicit" jagged Tensor inputs (i.e. dense Tensors with the first
    dimension == the JaggedIntVar.total_length() in the jagged Tensor
    inputs). Here we detect such implicit jagged Tensor inputs and replace
    the total_length: IntVar in the dense input's shape by the corresponding
    JaggedIntVar from the jagged input's shape. Importantly, this must be
    done before the mixed jagged / dense broadcasting takes place.
    """
    total_length_map = {}
    for tensor in inputs:
        if tensor.is_jagged():
            jagged_int_var = tensor._attrs["shape"][0]
            total_length = jagged_int_var.total_length()
            total_length_map[total_length] = jagged_int_var

    if total_length_map:
        # there are explicit jagged Tensors among the inputs:
        # we check if there are implict ones and make them explicit
        for tensor in inputs:
            shape = tensor._attrs["shape"]
            if not tensor.is_jagged() and shape and not isinstance(shape[0], IntImm):
                if shape[0] in total_length_map:
                    # the dense Tensor input's first dimension is the total_length
                    # dimension in the JaggedIntVar of one of the jagged Tensor
                    # inputs: we replace the dense Tensor input's first dimension
                    # by the corresponding JaggedIntVar, hence giving it a
                    # jagged Tensor semantics for further processing.
                    shape[0] = total_length_map[shape[0]]


def _broadcast_dense_shapes(shapes: List[List[IntVar]]) -> List[IntVar]:
    if len(shapes) == 1:
        return list(shapes[0])

    max_shape = None
    for shape in shapes:
        if max_shape is None:
            max_shape = list(shape)
        broadcastable, new_max_shape = shape_utils.get_broadcast_max_shape(
            max_shape, shape
        )
        if not broadcastable:
            raise ValueError(
                "Input shapes of the elementwise op are not compatible! "
                f"Shape1: {max_shape}, shape2: {shape}"
            )
        max_shape = new_max_shape

    return max_shape


def _broadcast_jagged_shapes(shapes: List[List[IntVar]]) -> List[IntVar]:
    if len(shapes) == 1:
        return list(shapes[0])

    rank = len(shapes[0])
    first_dim = shapes[0][0]
    for shape in shapes[1:]:
        other_first_dim = shape[0]
        if other_first_dim != first_dim:
            raise ValueError(
                "All jagged inputs of an elementwise op must "
                "have the same first dim (JaggedIntVar), but got "
                f"{first_dim} != {other_first_dim}"
            )
        other_rank = len(shape)
        if other_rank != rank:
            raise ValueError(
                "All jagged inputs of an elementwise op "
                "must have the same rank, but got "
                f"{rank} != {other_rank}"
            )

    suffix_shapes = [shape[1:] for shape in shapes]
    max_suffix_shape = suffix_shapes[0]
    for suffix_shape in suffix_shapes[1:]:
        broadcastable, new_max_shape = shape_utils.get_broadcast_max_shape(
            max_suffix_shape, suffix_shape
        )
        if not broadcastable:
            raise ValueError(
                "Jagged input suffix shapes of the elementwise op are not compatible! "
                f"Shape1: {max_suffix_shape}, shape2: {suffix_shape}"
            )
        max_suffix_shape = new_max_shape

    return [first_dim] + max_suffix_shape


def _broadcast_dense_and_jagged_shape(
    dense_shape: List[IntVar],
    jagged_shape: List[IntVar],
) -> List[IntVar]:
    jagged_first_dim = jagged_shape[0]
    jagged_suffix_shape = jagged_shape[1:]
    dense_suffix_shape = dense_shape[-len(jagged_suffix_shape) :]
    broadcastable, max_suffix_shape = shape_utils.get_broadcast_max_shape(
        jagged_suffix_shape, dense_suffix_shape
    )
    if not broadcastable:
        raise ValueError(
            "The suffix shapes of jagged and dense inputs of the elementwise op are not compatible! "
            f"Jagged suffix shape: {jagged_suffix_shape}, dense suffix shape: {dense_suffix_shape}"
        )

    if len(dense_shape) >= len(jagged_shape):
        dense_prefix_shape = dense_shape[: -len(dense_suffix_shape)]
        jagged_max_dense_prefix_shape = jagged_first_dim.get_max_dense_shape()
        if len(dense_prefix_shape) > len(jagged_max_dense_prefix_shape):
            raise ValueError(
                "The rank of dense inputs of an elementwise op can't be "
                "higher than the rank of the jagged inputs (when treating "
                "the jagged dims as separate dims)."
            )

        broadcastable, _ = shape_utils.get_broadcast_max_shape(
            jagged_max_dense_prefix_shape, dense_prefix_shape
        )
        if not broadcastable:
            raise ValueError(
                f"JaggedIntVar of the jagged inputs ({jagged_first_dim}) is not compatible "
                f"with the broadcasted prefix shape of the dense inputs ({dense_prefix_shape})."
            )

    return [jagged_first_dim] + max_suffix_shape


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

        _discover_implicit_jagged_inputs(args)

        dense_shapes = [arg._attrs["shape"] for arg in args if not arg.is_jagged()]
        jagged_shapes = [arg._attrs["shape"] for arg in args if arg.is_jagged()]

        max_dense_shape = _broadcast_dense_shapes(dense_shapes)
        if not jagged_shapes:
            return max_dense_shape

        max_jagged_shape = _broadcast_jagged_shapes(jagged_shapes)
        if not dense_shapes:
            return max_jagged_shape

        return _broadcast_dense_and_jagged_shape(max_dense_shape, max_jagged_shape)

    def __call__(self, *args: Tensor) -> Tensor:
        converted_args = []
        symbolic_args = []
        common_dtype = None
        assert len(args) > 0, "Elementwise ops must take at least one argument."
        for arg in args:
            if isinstance(arg, int) or isinstance(arg, float):
                converted_args.append(Tensor(shape=[], value=arg))
                symbolic_args.append(arg)
            elif isinstance(arg, IntVarTensor) and self._attrs["func"] == FuncEnum.SQRT:
                assert len(arg._attrs["int_var"]._attrs["values"]) == 1
                converted_args.append(
                    Tensor(shape=[], value=arg._attrs["int_var"]._attrs["values"][0])
                )
                symbolic_args.append(arg._attrs["int_var"].symbolic_value())
            elif isinstance(arg, Tensor):
                converted_args.append(arg)
                if common_dtype is None:
                    common_dtype = normalize_dtype(arg.dtype())
                elif normalize_dtype(arg.dtype()) != common_dtype:
                    raise NotImplementedError(
                        f"Type promotions are not supported; got dtype {arg.dtype()}, but expected {common_dtype}"
                    )
                symbolic_args.append(arg._attrs.get("symbolic_value", None))
            else:
                raise RuntimeError(
                    f"Unsupported data type {arg} in elementwise {self}!"
                )

        if common_dtype is None:
            # All inputs were constants. Just use fp16
            common_dtype = "float16"
        else:
            # Infer dtype for constant nums
            for arg in converted_args:
                if arg.is_a_const_num():
                    arg._attrs["dtype"] = common_dtype

        self._attrs["args"] = list(converted_args)
        self._attrs["inputs"] = [
            arg for arg in converted_args if not arg.is_a_const_num()
        ]
        self._set_depth()
        output_shape = self._infer_shapes(*converted_args)
        output = Tensor(output_shape, src_ops={self}, dtype=common_dtype)
        if self._attrs["func"] in INT_ELEMENTWISE_FUNC and None not in symbolic_args:
            output._attrs["symbolic_value"] = functools.reduce(
                INT_ELEMENTWISE_FUNC[self._attrs["func"]], symbolic_args
            )
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"func_enum": self._attrs["func"]}

    def replace_input_tensor(self, old_tensor, new_tensor) -> None:
        super().replace_input_tensor(old_tensor, new_tensor)
        self._attrs["args"] = [
            new_tensor if tensor is old_tensor else tensor
            for tensor in self._attrs["args"]
        ]

    def _args_for_pseudo_code(self):
        return [self._attrs["func"]]


# TODO: move it to math.py and update it to a function.
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


def _elementwise_func(func_enum: FuncEnum, *args: Tensor) -> Tensor:
    return elementwise(func_enum)(*args)


# Initialize OP_REGISTRY so that Tensor built-in functions can use.
for name, func_enum in FuncEnum.__members__.items():
    OP_REGISTRY[name] = functools.partial(_elementwise_func, func_enum)
