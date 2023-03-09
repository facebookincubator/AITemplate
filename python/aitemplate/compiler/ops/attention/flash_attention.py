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

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,W0223

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}total = {{x_dim0}};
{{indent}}{{dtype}}num_heads = {{x_dim2}};
{{indent}}{{dtype}}head_sizes = {{x_dim3}};
{{indent}}{{dtype}}NO = total;
{{indent}}{{dtype}}HO = num_heads;
{{indent}}{{dtype}}WO = head_sizes;
"""
)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
total == {{x_dim0}} && num_heads == {{x_dim2}} && head_sizes == {{x_dim3}}
"""
)


class flash_attention(Operator):
    r"""FlashAttention provides an implementation for fused
    multi-head attention module:

    .. math::
        \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK}{\sqrt(d)}) * V

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    """

    def __init__(self, batch_size, dropout, max_seq_len, causal) -> None:
        """Initialize attention module"""
        super().__init__()
        assert dropout == 0
        self._attrs["op"] = "flash_attention"
        self._attrs["has_profiler"] = False
        self._attrs["batch_size"] = batch_size
        self._attrs["dropout"] = dropout
        self._attrs["max_seq_len"] = max_seq_len
        self._attrs["seq_len"] = 512
        self._attrs["head_size"] = -1
        self._attrs["causal"] = causal
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
            x_dim3=x[3],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["HO"]),
            int(output["WO"]),
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

        output_shape = [
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
        ]
        return output_shape

    def __call__(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        """call the op

        Parameters
        ----------
        x : float16
            QKV tensor
            shape: (batch*seqlen, 3, num_heads, head_size)
        cu_seqlens : int
            seq lens tensor
            shape (batch_size + 1)

        Returns
        ----------
            Tensor
        """
        self._attrs["inputs"] = [x, cu_seqlens]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x, cu_seqlens)
        output = Tensor(output_shape, src_ops={self})

        batch_size = self._attrs["batch_size"]
        max_seq_len = self._attrs["max_seq_len"]
        total = x._attrs["shape"][0]._attrs["values"][0]
        num_heads = x._attrs["shape"][2]._attrs["values"][0]
        head_size = x._attrs["shape"][3]._attrs["values"][0]
        assert head_size in [8, 16, 32, 64, 128]
        self._attrs["head_size"] = head_size

        base_N = 256  # SM80
        if max_seq_len <= 128:
            seq_len = 128
        elif max_seq_len <= 256:
            seq_len = 256
        else:
            seq_len = ((max_seq_len + base_N - 1) // base_N) * base_N
        self._attrs["seq_len"] = seq_len

        self._attrs["workspace"] = (
            4 * num_heads * (total * head_size + batch_size * seq_len)
        )
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        target_attrs = ["batch_size", "dropout", "max_seq_len", "causal"]
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
