# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] nhwc 3 channel to 8 channel padding
"""
from typing import List

import jinja2

from .... import backend
from ....backend import registry
from ...base import IntImm, Operator, Tensor

# pylint: disable=C0103,W0221

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{% for dim in shape %}
{{indent}}{{dtype}}X_DIM{{loop.index - 1}} = {{dim}};
{% endfor %}
{{indent}}{{dtype}}Y_OUT_DIM = {{out_dim}};
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{% for dim in shape %}
{{indent}}{{dtype}}{{dim}} = X_DIM{{loop.index - 1}};
{% endfor %}
{{indent}}{{dtype}}{{last_dim}} = Y_OUT_DIM;
"""
)


class pad_last_dim(Operator):
    """[summary]

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self, ndim: int, out_dim: int):
        """[summary]"""
        super().__init__()
        self._attrs["op"] = "pad_last_dim"
        self._attrs["ndim"] = ndim
        self._attrs["out_dim"] = out_dim
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE

    def _infer_shapes(self, x: Tensor):
        """Infers shapes for pad_last_dim."""

        x_shape = x._attrs["shape"]
        ndim = len(x_shape)
        if self._attrs["out_dim"] <= max(x_shape[-1]._attrs["values"]):
            raise RuntimeError("Output of padded dim must be larger than original dim")
        if ndim != self._attrs["ndim"]:
            raise RuntimeError("Data/Op dims mismatch")
        if ndim > 4:
            raise NotImplementedError
        output_shape = list(x_shape)
        output_shape[-1] = IntImm(self._attrs["out_dim"])
        return output_shape

    def __call__(self, x: Tensor) -> List[Tensor]:
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]

        Returns
        -------
        List[Tensor]
            [description]
        """
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """[summary]

        Returns
        -------
        str
            [description]
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
            self.shape_eval_template,
            self.shape_save_template,
        )

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
        return func(self._attrs["name"])

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
        return func(self._attrs["name"])
