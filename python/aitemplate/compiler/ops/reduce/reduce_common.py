# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from typing import List

from .... import backend
from ....backend import registry
from ....utils.tensor_utils import wrap_dim
from ...base import IntImm, IntVar, Operator, Tensor
from ...tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0221


class reduce_base(Operator):
    """[summary]

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self, dim, keepdim=False, dtype=None) -> None:
        """[summary]

        Parameters
        ----------
        dim : int or tuple of python:ints
            [description] the dimension or dimensions to reduce
        keepdim : bool
            [description] keep the reduced dimensions if True, default is False
        dtype : optional str
            [description] the type of the return tensor. If it is not None,
                          the input tensor is casted to dtype before reduction.
        Raises
        ------
        RuntimeError : duplicate values in the dim list
            [description]
        """
        super().__init__()
        if isinstance(dim, int):
            dim = [dim]
        elif isinstance(dim, (list, tuple)):
            if not all(isinstance(x, int) for x in dim):
                raise RuntimeError("dim must be either int or a list/tuple of ints.")
            dim = list(dim)
        else:
            raise RuntimeError("dim must be either int or a list/tuple of ints.")
        dup_dims = {d for d in dim if dim.count(dim) > 1}
        if len(dup_dims) > 1:
            raise RuntimeError(
                "dim {d} appears multiple times in the list of dims".format(
                    d=dup_dims[0]
                )
            )
        self._attrs["op"] = "reduce"
        self._attrs["reduction_axes"] = dim
        self._attrs["keepdim"] = keepdim
        self._attrs["output_type"] = dtype
        self._attrs["has_profiler"] = False

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for reduce ops."""

        input_dims = x._attrs["shape"]
        reduction_axes = self._attrs["reduction_axes"]
        if self._attrs["keepdim"]:
            output_dims = [
                IntImm(1) if idx in set(reduction_axes) else d
                for idx, d in enumerate(input_dims)
            ]
        else:
            # out codegen for reduce ops doesn't rely on the output shape,
            # so it's safe to squeeze the output tensor shape here
            output_dims = [
                d for idx, d in enumerate(input_dims) if idx not in set(reduction_axes)
            ]
        return output_dims

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()
        reduction_axes = self._attrs["reduction_axes"]
        input_rank = len(x._attrs["shape"])
        self._attrs["reduction_axes"] = [
            wrap_dim(axis, input_rank) for axis in reduction_axes
        ]
        for axis in self._attrs["reduction_axes"]:
            if axis < 0 or axis >= input_rank:
                raise RuntimeError(
                    "invalid axis {a}, expected in a range [0, {r})".format(
                        a=axis, r=input_rank
                    )
                )
        output_shape = self._infer_shapes(x)
        output_type = self._attrs["output_type"]
        if output_type is None:
            output_type = x._attrs["dtype"]
        output = Tensor(output_shape, src_ops={self}, dtype=output_type)
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output

    def gen_function(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_decl(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_decl".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_call(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_call".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
