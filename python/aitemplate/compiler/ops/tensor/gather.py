# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from .... import backend
from ....backend import registry
from ...base import Operator, Tensor

# pylint: disable=C0103,W0221,W0102,W0223


class gather(Operator):
    """gather implementation

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self) -> None:
        """[summary]

        Parameters
        ----------

        """
        super().__init__()
        self._attrs["op"] = "gather"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor, dim: int, index: Tensor) -> Tensor:
        """[summary]

        Parameters
        ----------

        Returns
        -------
        """
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
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def _get_func(self, fmt_str):
        """[summary]

        Parameters
        ----------
        inputs : string
            [description] format string to create func_key for looking up func
                          from the registry

        Returns
        -------
        [type]
            [description]
        """
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)

    def gen_function_decl(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function_decl")
        return func(self._attrs)

    def gen_function_call(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
        """
        func = self._get_func("{target}.{op}.gen_function_call")
        return func(self._attrs)
