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
Flash attention.
"""
import itertools
from collections import OrderedDict
from typing import List

import jinja2
import numpy as np

from .... import backend
from ....backend import registry
from ....utils import shape_utils
from ...base import Operator, Tensor

# pylint: disable=C0103,W0221,W0102,W0223

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}B = {{x_dim0}};
{{indent}}{{dtype}}num_heads = {{x_dim1}};
{{indent}}{{dtype}}M = {{x_dim2}};
{{indent}}{{dtype}}Kv = {{x_dim3}};
"""
)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
batch_size == {{x_dim0}} && num_heads == {{x_dim1}} && seq_len == {{x_dim2}} && head_sizes == {{x_dim3}}
"""
)


class mem_eff_attention(Operator):
    r"""mem_eff_attention provides an implementation for fused
    multi-head attention module:

    .. math::
        \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK}{\sqrt(d)}) * V

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    """

    def __init__(self, causal, dropout=0) -> None:
        """Initialize attention module"""
        super().__init__()
        assert dropout == 0
        self._attrs["op"] = "mem_eff_attention"
        self._attrs["has_profiler"] = False
        self._attrs["dropout"] = dropout
        self._attrs["causal"] = causal
        self._attrs["head_size"] = -1
        self._attrs["workspace"] = 0
        self.exec_key_template = EXEC_KEY_TEMPLATE
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE

    def _infer_shape(self, x: List[int], w: List[int]):
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=w[3],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["B"]),
            int(output["M"]),
            int(output["num_heads"]),
            int(output["Kv"]),
        ]

    def _infer_shapes(self, x: Tensor, w: Tensor):
        """infer the output shape for attention module"""
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        w_shape = [var._attrs["values"][0] for var in w._attrs["shape"]]
        # run infer shape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape, w_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        batch_info = x._attrs["shape"][0]
        output_shape = [
            batch_info,
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]
        return output_shape

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """call the op

        Parameters
        ----------
        qkv : float16
            QKV tensor
            shape: (b, seqlen, num_heads, Kv)

        Returns
        ----------
            Tensor
        """

        head_size_v = v._attrs["shape"][3]._attrs["values"][0]
        self._attrs["head_size"] = head_size_v

        self._attrs["inputs"] = [q, k, v]
        self._set_depth()
        self._extract_exec_path(q)
        output_shape = self._infer_shapes(q, v)

        o_shape = [var._attrs["values"][-1] for var in output_shape]
        if o_shape[-1] > 128:
            self._attrs["workspace"] = 4 * np.prod(o_shape)
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        target_attrs = ["causal"]
        attr = {}

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr

    def _gen_exec_key(self, shape):
        """rendering shape info"""
        return self.exec_key_template.render(
            x_dim0=shape[0],
            x_dim1=shape[1],
            x_dim2=shape[2],
            x_dim3=shape[3],
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        self._attrs["exec_path"] = OrderedDict()
        for x_shape in x_shapes:
            key = self._gen_exec_key(x_shape)
            self._attrs["exec_path"][key] = ""

    def gen_function(self) -> str:
        """call backend functions"""
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
