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
Grouped GEMM Specialization: GEMM_RCR(A, B) + Bias
"""
from collections import OrderedDict
from typing import List

import jinja2

from aitemplate.compiler.base import ExecItem, Tensor
from aitemplate.compiler.ops.gemm_universal.gemm_rcr_bias import gemm_rcr_bias
from aitemplate.compiler.ops.gemm_universal.group_gemm_rcr import (
    group_gemm_rcr,
    SHAPE_EVAL_TEMPLATE,
)

from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


EXEC_KEY_TEMPLATE = jinja2.Template(
    """
{% for mnk in group_mnk %} {% if loop.index0 != 0 %} && {% endif %}
GROUP_{{loop.index0}}_M == {{mnk[0]}} &&
GROUP_{{loop.index0}}_N == {{mnk[1]}} &&
GROUP_{{loop.index0}}_K == {{mnk[2]}}
{% endfor %}
"""
)


class group_gemm_rcr_bias(group_gemm_rcr):
    """Grouped GEMM Specialization: GEMM_RCR(A, B) + Bias

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        # group 1
        A1 = torch.randn(M1, K1).cuda().half()
        B1 = torch.randn(N1, K1).cuda().half()
        Bias1 = torch.randn(N1).cuda().half()

        y1 = torch.nn.functional.linear(A1, B1, bias=Bias1)

        ...
        # group n
        An = torch.randn(Mn, Kn).cuda().half()
        Bn = torch.randn(Nn, Kn).cuda().half()
        Biasn = torch.randn(Nn).cuda().half()

        yn = torch.nn.functional.linear(An, Bn, bias=Biasn)
    """

    def __init__(self):
        super().__init__()
        self.shape_eval_template = SHAPE_EVAL_TEMPLATE
        self._attrs["op"] = "group_gemm_rcr_bias"

    def _extract_exec_path(self, dynamic_profiling_strategy=None):
        if dynamic_profiling_strategy is not None:
            # FIXME: Make group_gemm support dynamic_profiling_strategy.
            return

        # check batch dim same for each group
        batch_dim = self._attrs["inputs"][0]._attrs["shape"][0]
        for i in range(self._attrs["groups"]):
            if batch_dim != self._attrs["inputs"][i * 3]._attrs["shape"][0]:
                raise RuntimeError("Batch dim is different in groups")
        # for each batch create exec_path
        self._attrs["exec_path"] = OrderedDict()
        for m_value in batch_dim._attrs["values"]:
            group_mnk = []
            for i in range(self._attrs["groups"]):
                b = self._attrs["inputs"][i * 3 + 1]
                mnk = [m_value]
                mnk.append(b._attrs["shape"][0]._attrs["values"][0])
                mnk.append(b._attrs["shape"][1]._attrs["values"][0])
                group_mnk.append(mnk)
            exec_key = EXEC_KEY_TEMPLATE.render(group_mnk=group_mnk).replace("\n", "")
            self._attrs["exec_path"][exec_key] = ExecItem(
                profiling_key=exec_key,
                exec_cond=exec_key,
                algo="",
            )

    def input_a_accessors(self) -> List[TensorAccessor]:
        return group_gemm_rcr._one_input_accessors(
            self._attrs["input_accessors"], num_inputs_per_group=3, idx=0
        )

    def input_b_accessors(self) -> List[TensorAccessor]:
        return group_gemm_rcr._one_input_accessors(
            self._attrs["input_accessors"], num_inputs_per_group=3, idx=1
        )

    def input_bias_accessors(self) -> List[TensorAccessor]:
        return group_gemm_rcr._one_input_accessors(
            self._attrs["input_accessors"], num_inputs_per_group=3, idx=2
        )

    def __call__(self, operand_groups: List[List[Tensor]], output_stride_dim=None):
        # FIXME: when output_stride_dim is specified, we will concat the outputs of the
        # grouped gemm along the stride_dim axis. It's a temporary solution for
        # a pattern where the outputs of a grouped gemm can be concatenated
        # to form a single larger tensor. We will write a pass to detect such a
        # pattern automatically.
        self._attrs["inputs"] = []
        ret = []
        epilogue_alignment = 8
        for a, b, bias in operand_groups:
            op = gemm_rcr_bias()
            c = op(a, b, bias)
            c._attrs["src_ops"] = StableSet([self])
            a._attrs["dst_ops"].remove(op)
            b._attrs["dst_ops"].remove(op)
            bias._attrs["dst_ops"].remove(op)
            epilogue_alignment = min(
                op._attrs["epilogue_alignment"], epilogue_alignment
            )
            ret.append(c)
            self._attrs["inputs"].append(a)
            self._attrs["inputs"].append(b)
            self._attrs["inputs"].append(bias)
        self._set_depth()
        self._attrs["input_accessors"] = [
            TensorAccessor(a) for i, a in enumerate(self._attrs["inputs"])
        ]
        self._attrs["output_accessors"] = [TensorAccessor(c) for c in ret]
        self._attrs["groups"] = len(ret)
        if output_stride_dim is not None:
            # FIXME: replace this manual concat with an automated pass
            if output_stride_dim != 1:
                raise RuntimeError(
                    "only support cases where output_stride_dim equals to 1"
                )
            self._attrs["output_stride_dim"] = output_stride_dim
            ret = self._concat_strided_outputs(ret, output_stride_dim)
            self._attrs["outputs"] = [ret]
        else:
            self._attrs["outputs"] = ret
        self._attrs["epilogue_alignment"] = epilogue_alignment
        self._extract_exec_path()
        # This is a lazy way to allocate space for args
        # Reserve 12 * 4 * len(groups) byte for each field
        # 12 is read of sizeof(GemmCoord)
        # problem_sizes_device
        # ptrA/B/C/D
        # lda/b/c/d
        # problem_sizes_device: N * GemmCoord -> N * 3 * sizeof(int64_t) ~ 32 * N
        # ptrA/B/C/D: N * sizeof(half*) ~ N * 8 for each
        # lda/b/c/d: N * sizeof(int64_t) ~ N * 8 for each
        # total: N * 8 * 4 + N * 8 * 4 + N * 8 * 4
        # total: 3 * 32 * N
        args_size = 96 * self._attrs["groups"]
        self._attrs["unique_workspace"] = args_size
        return ret
