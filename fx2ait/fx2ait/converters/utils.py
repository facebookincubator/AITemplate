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
import math
import operator
from typing import Any, Callable, Dict, List, Tuple, Union

from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor

from aitemplate.compiler.public import elementwise, FuncEnum, Tensor as AITTensor
from torch.fx.node import Argument


def get_positive_dim(dim: int, dim_size: int) -> int:
    if dim < 0:
        return dim % dim_size
    return dim


def create_reduce_op(
    op_type: Any, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: str
) -> AITTensor:
    input_val = kwargs["input"]
    # TODO: remove once multiple reduction axes are supported
    dims = kwargs.get("dim", None)
    if dims is None:
        dims = list(range(len(input_val.shape())))
    if len(dims) < 1:
        raise ValueError("No dims to reduce on")
    dim = dims[0]
    keepdim = False if "keepdim" not in kwargs else kwargs["keepdim"]
    sum_val = op_type(dim=dim, keepdim=keepdim)(input_val)

    if len(dims) > 1:
        new_kwargs = {"input": sum_val, "dims": dims[1:]}
        return create_reduce_op(op_type, args, new_kwargs, name)

    return sum_val


def create_binary_op(
    op_type: FuncEnum,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> AITTensor:
    lhs = kwargs["input"]
    if not isinstance(lhs, (AITTensor, float, int)):
        raise RuntimeError(f"Unexpected left operand {type(lhs)} on {name}: {lhs}")

    rhs = kwargs["other"]
    if not isinstance(rhs, (AITTensor, float, int)):
        raise RuntimeError(f"Unexpected right operand {type(rhs)} on {name}: {rhs}")

    lhs_is_constant, lhs_constant = try_get_constant_num(lhs)
    rhs_is_constant, rhs_constant = try_get_constant_num(rhs)
    if lhs_is_constant and rhs_is_constant:
        res = get_python_op_from_ait_constant_elementwise_op(op_type)(
            lhs_constant, rhs_constant
        )
        return res

    return elementwise(op_type)(lhs, rhs)


def create_unary_op(
    op_type: FuncEnum,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> AITTensor:
    input = kwargs["input"] if "input" in kwargs else args[0]
    if not isinstance(input, (AITTensor, float, int)):
        raise RuntimeError(f"Unexpected left operand {type(input)} on {name}: {input}")

    input_is_constant, input_constant = try_get_constant_num(input)
    if input_is_constant:
        res = get_python_op_from_ait_constant_elementwise_op(op_type)(input_constant)
        return res

    return elementwise(op_type)(input)


def try_get_constant_num(arg: Any) -> (bool, Any):
    if isinstance(arg, (float, int)):
        return (True, arg)
    elif isinstance(arg, IntImm):
        return (True, arg.value())
    elif isinstance(arg, IntVarTensor):
        var = arg._attrs["int_var"]
        return try_get_constant_num(var)
    else:
        return (False, None)


def get_python_op_from_ait_constant_elementwise_op(
    op_type: FuncEnum,
) -> Callable[[Any, Any], Any]:
    if op_type == FuncEnum.ADD:
        return operator.add
    elif op_type == FuncEnum.MUL:
        return operator.mul
    elif op_type == FuncEnum.SUB:
        return operator.sub
    elif op_type == FuncEnum.DIV:
        return operator.truediv
    elif op_type == FuncEnum.SQRT:
        return math.sqrt
    else:
        raise RuntimeError(f"{op_type} is not supported yet!")


def identical_elem_tuple_to_int(param):
    """
    Convert tuples with all the same int elem to
    a single int (ex. (3, 3, 3) --> 3)
    """
    if isinstance(param, int):
        return param

    if not isinstance(param, (list, tuple)) or not all(x == param[0] for x in param):
        raise RuntimeError(f"AIT supports square param values only, but got {param}")
    return param[0]


def nchw2nhwc(shape: List[Union[int, IntVar]]) -> List[Union[int, IntVar]]:
    return [shape[0], shape[2], shape[3], shape[1]]


def ncdhw2ndhwc(shape: List[Union[int, IntVar]]) -> List[Union[int, IntVar]]:
    return [shape[0], shape[2], shape[3], shape[4], shape[1]]


# TODO:  This is a hack to workaround AIT's dynamic shape requirement.
# Detailed explanation can be found in D41743385 (aten2ait) D41974191(fx2ait).
# We will throw this one after AIT provides vanilla support.
def unify_dynamic_shape_name(input_val, weight):
    input_shape = input_val.shape()
    weight_shape = weight.shape()
    if len(input_shape) == len(weight_shape):
        for a, b in zip(input_shape, weight_shape):
            if a._attrs["values"] == b._attrs["values"]:
                if a._attrs["name"] is None:
                    a._attrs["name"] = b._attrs["name"]
                elif b._attrs["name"] is None:
                    b._attrs["name"] = a._attrs["name"]
    return input_shape, weight_shape
