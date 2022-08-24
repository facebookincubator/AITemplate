# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
permute(2, 1, 0) op
Swap the dimension of dim0 and dim2 of input 3d tensor.
"""
from typing import List

from aitemplate.backend import registry

from .... import backend
from ...base import Operator, Tensor

# pylint: disable=C0103,W0221


class permute210(Operator):
    """
    Change the dimension of dim0 and dim2 of input 3d tensor.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "permute210"

    def _infer_shapes(self, x: Tensor):
        """Infers shapes for permute210.

        Parameters
        ----------
        x : Tensor

        Returns
        ------
        Tensor
            Inferred output 3d tensor with input shape.
            Because its a permute210 operation,
            Out.d0=In.d2, Out.d2=In.d0.
        """
        x_shape = x._attrs["shape"]
        return [x_shape[2], x_shape[1], x_shape[0]]

    def __call__(self, x: Tensor) -> List[Tensor]:
        """
        Return the output tensor of permute210

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor
            Generate output tensors of function calls.
            In permute210, its a 3d tensor with d2,d1,d0 of
            input Tensor.
        """
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """
        Generate function body

        Returns
        -------
        str
           The function body string
        """
        target = backend.target.Target.current()
        template_path = target.template_path()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            template_path,
        )

    def gen_function_decl(self) -> str:
        """
        Generate function declaration.

        Returns
        -------
        str
            The function declaration string
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_decl".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs["name"])

    def gen_function_call(self) -> str:
        """
        Generate function call.

        Returns
        -------
        str
            The function call string
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_call".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs["name"])
