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
upsampling2d.
"""
import itertools
import logging
import re
from collections import OrderedDict
from typing import List

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,R1732,W0613
logging.basicConfig(level=logging.INFO)


SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}CI = {{x_dim3}};
{{indent}}{{dtype}}CO = {{x_dim3}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}HO = HI * {{scale_factor}};
{{indent}}{{dtype}}WO = WI * {{scale_factor}};
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = HO;
{{indent}}{{y_dim2}} = WO;
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class upsampling2d_base(Operator):
    """
    Upsamples a given multi-channel 2D (spatial) data.

    * :attr:`scale_factor` (float): multiplier for spatial size.

    * :attr:`mode` (str): the upsampling algorithm: one of ``'nearest'``,
      ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
      Currently we support ``'bilinear'`` and  ``'nearest'`` mode.

    Args:
        input (Tensor [N, H, W, C]): the input data.

    Return:
        Tensor [N, H_out, W_out, C].
    """

    def __init__(self, scale_factor, mode) -> None:
        super().__init__()
        self._attrs["op"] = "upsampling2d"
        self._attrs["scale_factor"] = scale_factor
        self._attrs["mode"] = mode
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _infer_shape(self, x: List[int]):
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            scale_factor=self._attrs["scale_factor"],
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

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape: List[int]):
        return self.exec_key_template.render(
            x_dim0=shape[0], x_dim1=shape[1], x_dim2=shape[2], x_dim3=shape[3]
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        self._attrs["exec_path"] = OrderedDict()
        self._attrs["exec_path"]["true"] = ""

    def _signature(self):
        signature = "upsampling2d: S=[{s}], M=[{m}]]".format(
            s=self._attrs["scale_factor"], m=self._attrs["mode"]
        )
        return signature

    def __call__(self, x: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {
            "mode": self._attrs["mode"],
            "scale_factor": self._attrs["scale_factor"],
        }

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
            self.exec_cond_template,
            self.shape_eval_template,
            self.shape_save_template,
        )
