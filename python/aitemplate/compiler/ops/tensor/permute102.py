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
Permute(1, 0, 2) op.
Change the dimension of dim0 and dim1 of input 3d tensor.
"""
from typing import List

import jinja2

from aitemplate.backend import registry

from .... import backend
from ...base import IntVar, Operator, Tensor

# pylint: disable=C0103,W0221

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}X_DIM0 = {{x_dim0}};
{{indent}}{{dtype}}X_DIM1 = {{x_dim1}};
{{indent}}{{dtype}}X_DIM2 = {{x_dim2}};
{{indent}}{{dtype}}Y_DIM0 = X_DIM1;
{{indent}}{{dtype}}Y_DIM1 = X_DIM0;
{{indent}}{{dtype}}Y_DIM2 = X_DIM2;
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = Y_DIM0;
{{indent}}{{y_dim1}} = Y_DIM1;
{{indent}}{{y_dim2}} = Y_DIM2;
"""
)


class permute102(Operator):
    """
    Permutes the input 3d tensor from (B, N, M) to (N, B, M).

    Args:
        input (Tensor[B, N, M]): the source tensor with 3 dimensions

    Returns:
        output (Tensor[N, B, M]): the destination tensor

    Example:

        .. highlight:: python
        .. code-block:: python

            X = Tensor(shape=[2, 384, 262], name="X", is_input=True)
            Y = ops.permute102()(X)
            y_shape = [d._attrs["values"][0] for d in Y.shape()]
            print(y_shape)

            Outs:
            [384, 2, 262]

    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "permute102"
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE

    def _infer_shape(self, x: List[int]):
        """
        Parameters
        ----------
        x : List[int]

        Returns
        -------
        List[int]
            Deduce output dimension based on SHAPE_ASSIGNMENT_TEMPLATE.
        """
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [int(output["Y_DIM0"]), int(output["Y_DIM1"]), int(output["Y_DIM2"])]

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for permute021."""

        x_shape = x._attrs["shape"]
        return [x_shape[1], x_shape[0], x_shape[2]]

    def __call__(self, x: Tensor) -> List[Tensor]:
        """
        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor
            Generate output tensors of function calls.
            In permute102, its a 3d tensor with d1,d0,d2 of
            input Tensor.
        """
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """Generate function body."""
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
