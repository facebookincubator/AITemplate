# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] nhwc 3 channel to 8 channel padding
"""
import itertools
from typing import List

import jinja2

from .... import backend
from ....backend import registry
from ....utils import shape_utils
from ...base import Operator, Tensor

# pylint: disable=C0103,W0221

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}HO = HI;
{{indent}}{{dtype}}WO = WI;
{{indent}}{{dtype}}CO = 8;
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = HO;
{{indent}}{{y_dim2}} = WO;
"""
)


class nhwc3to8(Operator):
    """[summary]

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(self):
        """[summary]"""
        super().__init__()
        self._attrs["op"] = "nhwc3to8"
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE

    def _infer_shape(self, x: List[int]):
        """[summary]

        Parameters
        ----------
        x : List[int]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor):
        """[summary]

        Parameters
        ----------
        x : Tensor
            [description]
        w : Tensor
            [description]

        Returns
        -------
        [type]
            [description]
        """
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]
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
