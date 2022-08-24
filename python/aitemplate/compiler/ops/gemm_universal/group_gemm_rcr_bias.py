# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
from collections import OrderedDict
from typing import List

import jinja2

from ...base import ExecItem
from ...tensor_accessor import TensorAccessor
from ...transform.refine_graph import same_int_var
from .gemm_rcr_bias import gemm_rcr_bias
from .group_gemm_rcr import group_gemm_rcr, SHAPE_EVAL_TEMPLATE

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
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
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
            if not same_int_var(
                batch_dim, self._attrs["inputs"][i * 3]._attrs["shape"][0]
            ):
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

    def __call__(self, operand_groups, output_stride_dim=None):
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
            c._attrs["src_ops"] = [self]
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
