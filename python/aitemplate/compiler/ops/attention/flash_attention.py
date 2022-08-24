# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import itertools
from collections import OrderedDict
from typing import List

import jinja2

from .... import backend
from ....backend import registry
from ....utils import shape_utils
from ...base import IntImm, Operator, Tensor

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
    """flash_attention provided a implementation for fused
    multi-head attention module:
    Attention(Q, K, V) = softmax(QK / sqrt(d)) * V
    MultiHead(Q, K, V) = Concat(head1,... headn) * W
        where head(i) = Attention(QWi, KWi, VWi)
    """

    def __init__(self, batch_size, dropout, max_seq_len, causal) -> None:
        """initilize attention module"""
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

        o_tmp = Tensor(
            [IntImm(total), IntImm(num_heads), IntImm(head_size)],
            dtype="float32",
            dst_ops={self},
        )
        softmax_lse = Tensor(
            [IntImm(batch_size), IntImm(num_heads), IntImm(seq_len)],
            dtype="float32",
            dst_ops={self},
        )

        self._attrs["inputs"].append(o_tmp)
        self._attrs["inputs"].append(softmax_lse)
        self._attrs["outputs"] = [output]
        return output

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

    def gen_function_decl(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_decl".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_call(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_call".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
