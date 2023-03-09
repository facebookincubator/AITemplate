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
Pad last dimension.
"""
from typing import List

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor

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
    """Pad the last dimension of the input data to the specified length."""

    def __init__(self, ndim: int, out_dim: int):
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
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"ndim": self._attrs["ndim"], "out_dim": self._attrs["out_dim"]}

    def gen_function(self) -> str:
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
