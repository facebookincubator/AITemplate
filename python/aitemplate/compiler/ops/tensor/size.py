# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Op to return the size of a tensor.
"""
from typing import List, Union

from aitemplate import backend

from aitemplate.backend import registry

from ...base import IntImm, IntVar, IntVarTensor, Operator, Tensor

# pylint: disable=C0103,W0221,R1732,W0613


class size(Operator):
    """
    Returns the size of the input tensor. If dim is not specified, the returned value is
    the same as tensor.shape(). If dim is specified, returns an int holding the size of
    that dimension.

    This op doesn't generate any code.
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "size"
        self._attrs["has_profiler"] = False

    def _copy_construct_IntVar_IntImm(self, var: IntVar) -> IntVar:
        if isinstance(var, IntImm):
            return IntImm(var._attrs["values"][0], var._attrs["name"], src_ops=[self])
        assert isinstance(var, IntVar)
        return IntVar(var._attrs["values"], var._attrs["name"], src_ops=[self])

    def __call__(self, x: Tensor, dim: int = None) -> Union[List[IntVar], IntVar]:
        self._attrs["inputs"] = [x]
        self._set_depth()
        if isinstance(dim, int):
            var = x._size(dim)
            output = IntVarTensor(var, src_ops={self})
            self._attrs["outputs"] = [output]
        else:
            output = [IntVarTensor(var, src_ops={self}) for var in x.shape()]
            self._attrs["outputs"] = output
        return output

    def gen_function(self) -> str:
        """call backend function"""
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_decl(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_decl".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_call(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_call".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
