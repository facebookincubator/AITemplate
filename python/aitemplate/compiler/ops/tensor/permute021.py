# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
permute(0, 2, 1) op
"""
from typing import List

import jinja2

from .... import backend
from ....backend import registry
from ...base import IntVar, Operator, Tensor

# pylint: disable=C0103,W0221

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}X_DIM0 = {{x_dim0}};
{{indent}}{{dtype}}X_DIM1 = {{x_dim1}};
{{indent}}{{dtype}}X_DIM2 = {{x_dim2}};
{{indent}}{{dtype}}Y_DIM0 = X_DIM0;
{{indent}}{{dtype}}Y_DIM1 = X_DIM2;
{{indent}}{{dtype}}Y_DIM2 = X_DIM1;
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = Y_DIM0;
{{indent}}{{y_dim1}} = Y_DIM1;
{{indent}}{{y_dim2}} = Y_DIM2;
"""
)


class permute021(Operator):
    """[summary]

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self):
        """[summary]"""
        super().__init__()
        self._attrs["op"] = "permute021"
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for permute021."""

        x_shape = x._attrs["shape"]
        return [x_shape[0], x_shape[2], x_shape[1]]

    def __call__(self, x: Tensor) -> List[Tensor]:
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
