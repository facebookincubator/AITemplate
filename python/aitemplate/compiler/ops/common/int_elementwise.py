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
Int elementwise operator definition, for integer calcuation on tensor dimensions.
"""
import functools
from functools import reduce

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntVarTensor, Operator, Tensor
from aitemplate.compiler.op_registry import OP_REGISTRY
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,C0301,W0223,R1724

INT_ELEMENTWISE_FUNC = {
    FuncEnum.MUL: lambda x, y: x * y,
    FuncEnum.DIV: lambda x, y: x / y,
    FuncEnum.SUB: lambda x, y: x - y,
    FuncEnum.ADD: lambda x, y: x + y,
}


class int_elementwise(Operator):
    """int elementwise operator definition."""

    def __init__(self, func_enum: FuncEnum) -> None:
        """
        Parameters
        ----------
        func_enum : the underlying function enum.
        """

        super().__init__()
        self._attrs["op"] = "int_elementwise"
        if func_enum not in INT_ELEMENTWISE_FUNC:
            raise RuntimeError(f"Not such FuncEnum {func_enum} in int_elementwise!")
        self._attrs["func"] = func_enum
        self._attrs["has_profiler"] = False

    def __call__(self, *args: IntVarTensor) -> Tensor:
        int_vars = []
        for arg in args:
            if isinstance(arg, IntVarTensor):
                int_vars.append(arg._attrs["int_var"])
            else:
                raise RuntimeError(
                    f"Unsupported data type {arg} in elementwise {self}!"
                )
        max_vars = [max(v._attrs["values"]) for v in int_vars]
        min_vars = [min(v._attrs["values"]) for v in int_vars]
        sym_vars = [v._attrs["symbolic_value"] for v in int_vars]
        assert len(max_vars) == len(min_vars) and len(max_vars) >= 2
        values = []
        if self._attrs["func"] == FuncEnum.MUL:
            values += [reduce(lambda x, y: x * y, lis) for lis in [min_vars, max_vars]]
            sym_values = reduce(INT_ELEMENTWISE_FUNC[FuncEnum.MUL], sym_vars)
        elif self._attrs["func"] == FuncEnum.ADD:
            values += [reduce(lambda x, y: x + y, lis) for lis in [min_vars, max_vars]]
            sym_values = reduce(INT_ELEMENTWISE_FUNC[FuncEnum.ADD], sym_vars)
        elif self._attrs["func"] == FuncEnum.SUB:
            inp_range = [(a, b) for a, b in zip(min_vars, max_vars)]
            # For an inputs of range [(4,9), (1,8)],
            # i.e. (4,9) is the range for first dim and (1,8) for the second
            # we want max(4,9)-min(1,8) as the new upper bound and
            # min(4,9)-max(1,8) as the new lower bound.
            # however, range should be larger than 1, thus we have max(1, min(4,9)-max(1,8))
            for i, b in enumerate(inp_range):
                if i == 0:
                    a = b
                else:
                    lower_bound = max(min(a[0], a[1]) - max(b[0], b[1]), 1)
                    upper_bound = max(a[0], a[1]) - min(b[0], b[1])
                    if upper_bound <= 0:
                        raise RuntimeError(
                            f"Subtracting Tensor with shape {b} from Tensor with shape {a} "
                            f"is invalid. Cannot deduce a valid bound."
                        )
                    a = (lower_bound, upper_bound)
            values = list(a)
            sym_values = reduce(INT_ELEMENTWISE_FUNC[FuncEnum.SUB], sym_vars)
        elif self._attrs["func"] == FuncEnum.DIV:  # floordiv
            inp_range = [(a, b) for a, b in zip(min_vars, max_vars)]
            # For an inputs of range [(4,9), (1,8)],
            # i.e. (4,9) is the range for first dim and (1,8) for the second
            # we want max(4,9)/min(1,8) as the new upper bound and
            # min(4,9)/max(1,8) as the new lower bound.
            # however, range should be larger than 1, thus we have max(1, min(4,9)/max(1,8))
            for i, b in enumerate(inp_range):
                if i == 0:
                    a = b
                else:
                    lower_bound = max(int(min(a[0], a[1]) / max(b[0], b[1])), 1)
                    upper_bound = int(max(a[0], a[1]) / min(b[0], b[1]))
                    if upper_bound <= 0:
                        raise RuntimeError(
                            f"Dividing Tensor with shape {b} from Tensor with shape {a} "
                            f"is invalid. Cannot deduce a valid bound."
                        )
                    a = (lower_bound, upper_bound)
            values = list(a)
            sym_values = reduce(INT_ELEMENTWISE_FUNC[FuncEnum.DIV], sym_vars)
        else:
            raise RuntimeError(f"Unsupported calculation type {self._attrs['func']}!")
        dim = shape_utils.gen_int_var_min_max(values, symbolic_value=sym_values)
        for arg, iv in zip(args, int_vars):
            arg._attrs["int_var"] = iv
            assert not arg.is_a_const_num(), f"{arg} cannot be constant"
        self._attrs["args"] = args
        self._attrs["inputs"] = list(args)
        self._set_depth()
        output = IntVarTensor(dim, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"func_enum": self._attrs["func"]}

    def _args_for_pseudo_code(self):
        return [self._attrs["func"]]

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)


def _int_elementwise_func(func_enum: FuncEnum, *args: IntVarTensor) -> Tensor:
    return int_elementwise(func_enum)(*args)


# Initialize OP_REGISTRY so that Tensor built-in functions can use.
for name, func_enum in FuncEnum.__members__.items():
    OP_REGISTRY["INT_" + name] = functools.partial(_int_elementwise_func, func_enum)
