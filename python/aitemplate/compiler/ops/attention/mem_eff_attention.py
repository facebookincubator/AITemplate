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
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import jinja2
import numpy as np

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntVar, Operator, Tensor
from aitemplate.utils import shape_utils

_LOGGER = logging.getLogger(__name__)

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

    def __init__(
        self,
        causal,
        dropout=0,
        variable_seq_length_kv=False,
        variable_seq_length_q=False,
        use_grouped_fmha=False,
    ) -> None:
        """Initialize attention module"""
        super().__init__()
        assert dropout == 0
        self._attrs["op"] = "mem_eff_attention"
        self._attrs["has_profiler"] = False
        self._attrs["dropout"] = dropout
        self._attrs["causal"] = causal
        self._attrs["variable_seq_length_kv"] = variable_seq_length_kv
        self._attrs["variable_seq_length_q"] = variable_seq_length_q
        self._attrs["head_size"] = -1
        self._attrs["workspace"] = 0
        self._attrs["use_grouped_fmha"] = use_grouped_fmha
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

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        lengths_kv: Optional[Tensor] = None,
        lengths_q: Optional[Tensor] = None,
    ) -> Tensor:
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
        if self._attrs["variable_seq_length_kv"]:
            assert lengths_kv is not None
            self._attrs["inputs"].append(lengths_kv)
        if self._attrs["variable_seq_length_q"]:
            assert lengths_q is not None
            self._attrs["inputs"].append(lengths_q)
        self._set_depth()
        self._extract_exec_path(q)
        output_shape = self._infer_shapes(q, v)

        required_workspace_size = self._compute_required_workspace(
            output_shape, q._attrs["shape"], k._attrs["shape"]
        )
        self._attrs["workspace"] = required_workspace_size
        _LOGGER.debug(f"Required workspace size: {required_workspace_size}")
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        return output

    def _compute_required_workspace(
        self,
        output_shape: Tuple[IntVar, IntVar, IntVar, IntVar],
        q_shape: Tuple[IntVar, IntVar, IntVar, IntVar],
        k_shape: Tuple[IntVar, IntVar, IntVar, IntVar],
    ) -> int:
        """
        Compute workspace size required for attention op.
        """
        is_float32 = self._attrs["inputs"][0]._attrs["dtype"] not in [
            "float16",
            "bfloat16",
        ]

        o_shape = [var._attrs["values"][-1] for var in output_shape]
        # We need a separate buffer of output accumulation
        # - when the intermediate output can't fit into the register file.
        # - when the accumulation type (float) is different from the output type.
        # See https://github.com/NVIDIA/cutlass/blob/209faf7b94ce4ba573d27389fb643962e75d0581/examples/41_fused_multi_head_attention/fmha_grouped.h#L79-L95
        needs_output_accum_buffer = (o_shape[-1] > 128) or not is_float32
        if needs_output_accum_buffer:  # Needs output accumulator buffer
            size_of_accum_element = 4  # Accumulation is always in float
            accu_size = size_of_accum_element * np.prod(o_shape)
        else:
            accu_size = 0

        # The backend which uses kernel_forward.h only needs accumulator buffer
        if not self._attrs["use_grouped_fmha"]:
            return accu_size

        # Number of problems is batch_size * num_heads
        problem_count = q_shape[0].upper_bound() * q_shape[1].upper_bound()

        size_of_int = 4
        size_of_int64 = 8
        # GEMM size is specified by 3 ints: m, n, k
        size_of_gemm_coord = 3 * size_of_int

        # There are two GEMM sizes for each problem, corresponding to 2 matrix
        # multiplications in attention
        problem_sizes_size = 2 * size_of_gemm_coord * problem_count

        # For each problem, need space for leading dimensions of 5 matrices:
        # Q, K, V, O. Leading dimensions are in int64.
        ld_sizes = 4 * size_of_int64 * problem_count

        # For each problem, pointers to 5 matrices: Q, K, V, O, O_accum
        size_of_ptr = 8  # 64-bit arch
        ptrs_sizes = 5 * size_of_ptr * problem_count
        total_size = problem_sizes_size + accu_size + ld_sizes + ptrs_sizes
        return total_size

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
        self._attrs["arch"] = target._arch
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
