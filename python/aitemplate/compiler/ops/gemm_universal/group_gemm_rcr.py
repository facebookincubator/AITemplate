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
Grouped GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
"""
import logging
import re
from collections import OrderedDict
from typing import List

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.compiler.base import ExecItem, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common
from aitemplate.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr
from aitemplate.compiler.ops.tensor import concatenate

from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


_LOGGER = logging.getLogger(__name__)

SHAPE_EVAL_TEMPLATE = jinja2.Template(
    """
{% for operand_dim in group_operand_dims %}
{% set output_addr = output_addr_cals[loop.index - 1] %}
{% set input_a_addr = input_a_addr_cals[loop.index - 1] %}
{{indent}}{{dtype}}GROUP_{{loop.index0}}_AM = {{operand_dim[0]}};
{{indent}}{{dtype}}GROUP_{{loop.index0}}_AK = {{operand_dim[1]}};
{{indent}}{{dtype}}GROUP_{{loop.index0}}_BN = {{operand_dim[2]}};
{{indent}}{{dtype}}GROUP_{{loop.index0}}_BK = {{operand_dim[3]}};
{{indent}}{{dtype}}GROUP_{{loop.index0}}_CM = GROUP_{{loop.index0}}_AM;
{{indent}}{{dtype}}GROUP_{{loop.index0}}_CN = GROUP_{{loop.index0}}_BN;

{{input_a_addr}}
{{output_addr}}

{{indent}}{{dtype}}GROUP_{{loop.index0}}_M = {{operand_dim[0]}};
{{indent}}{{dtype}}GROUP_{{loop.index0}}_K = {{operand_dim[1]}};
{{indent}}{{dtype}}GROUP_{{loop.index0}}_N = {{operand_dim[2]}};
{% endfor %}
{% for operand_dim in group_operand_dims %}
{{indent}}{{operand_dim[4]}} = GROUP_{{loop.index0}}_M;
{{indent}}{{operand_dim[5]}} = GROUP_{{loop.index0}}_N;
{% endfor %}
"""
)


EXEC_KEY_TEMPLATE = jinja2.Template(
    """
{% for mnk in group_mnk %} {% if loop.index0 != 0 %} && {% endif %}
GROUP_{{loop.index0}}_M == {{mnk[0]}} &&
GROUP_{{loop.index0}}_N == {{mnk[1]}} &&
GROUP_{{loop.index0}}_K == {{mnk[2]}}
{% endfor %}
"""
)


class group_gemm_rcr(common.gemm):
    """Grouped GEMM Specialization: GEMM_RCR(A, B)

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        # group 1
        A1 = torch.randn(M1, K1).cuda().half()
        B1 = torch.randn(N1, K1).cuda().half()

        y1 = torch.nn.functional.linear(A1, B1)

        ...
        # group n
        An = torch.randn(Mn, Kn).cuda().half()
        Bn = torch.randn(Nn, Kn).cuda().half()

        yn = torch.nn.functional.linear(An, Bn)
    """

    def __init__(self):
        super().__init__()
        self.shape_eval_template = SHAPE_EVAL_TEMPLATE
        self._attrs["op"] = "group_gemm_rcr"
        # this is a state flag will be codegen
        self._attrs["int_state_flag"] = 0

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k, self._attrs["inputs"][0].dtype())

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _invert_exec_key(self, key):
        tmp = re.findall(r"==\s*(\d+)", key)
        return [int(x) for x in tmp]

    def _extract_exec_path(self, dynamic_profiling_strategy=None):
        # FIXME: Make this API properly support dynamic_profiling_strategy.
        if dynamic_profiling_strategy is not None:
            return

        # check batch dim same for each group
        batch_dim = self._attrs["inputs"][0]._attrs["shape"][0]
        for i in range(self._attrs["groups"]):
            if batch_dim != self._attrs["inputs"][i * 2]._attrs["shape"][0]:
                raise RuntimeError(
                    "Batch dim is different in groups. Inputs: {}".format(
                        self._attrs["inputs"]
                    )
                )
        # for each batch create exec_path
        self._attrs["exec_path"] = OrderedDict()
        for m_value in batch_dim._attrs["values"]:
            group_mnk = []
            for i in range(self._attrs["groups"]):
                b = self._attrs["inputs"][i * 2 + 1]
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

    def _gen_profile_cmd(self, profiler_prefix, cfg, exec_key):
        def fbuild_cmd(exec_key):
            mnk_flat = self._invert_exec_key(exec_key)
            cmd = []
            cmd.append(self._attrs["groups"])
            cmd.extend(mnk_flat)
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)

    def _concat_strided_outputs(self, outputs, output_stride_dim):
        """a temporary function to concatenate strided outputs"""
        cat_op = concatenate()
        cat_output = cat_op(outputs, dim=output_stride_dim)
        cat_output._attrs["src_ops"] = StableSet([self])
        offset = 0
        for idx, output_tensor in enumerate(outputs):
            self._attrs["output_accessors"][idx].update_base_tensor(
                cat_output, output_stride_dim, offset
            )
            offset += output_tensor._attrs["shape"][output_stride_dim]._attrs["values"][
                0
            ]
            from aitemplate.compiler.transform import transform_utils

            transform_utils.remove_tensor_from_sorted_graph(output_tensor)
        return cat_output

    @staticmethod
    def _one_input_accessors(
        input_accessors: List[TensorAccessor], num_inputs_per_group: int, idx: int
    ) -> List[TensorAccessor]:
        return [
            a for i, a in enumerate(input_accessors) if i % num_inputs_per_group == idx
        ]

    def input_a_accessors(self) -> List[TensorAccessor]:
        return self._one_input_accessors(
            self._attrs["input_accessors"], num_inputs_per_group=2, idx=0
        )

    def input_b_accessors(self) -> List[TensorAccessor]:
        return self._one_input_accessors(
            self._attrs["input_accessors"], num_inputs_per_group=2, idx=1
        )

    def __call__(self, operand_groups: List[List[Tensor]], output_stride_dim=None):
        # FIXME: when output_stride_dim is specified, we will concat the outputs of the
        # grouped gemm along the output_stride_dim axis. It's a temporary solution for
        # a pattern where the outputs of a grouped gemm can be concatenated
        # to form a single larger tensor. We will write a pass to detect such a
        # pattern automatically.
        self._attrs["inputs"] = []
        ret = []
        epilogue_alignment = 8
        for a, b in operand_groups:
            op = gemm_rcr()
            c = op(a, b)
            c._attrs["src_ops"] = StableSet([self])
            a._attrs["dst_ops"].remove(op)
            b._attrs["dst_ops"].remove(op)
            epilogue_alignment = min(
                op._attrs["epilogue_alignment"], epilogue_alignment
            )
            ret.append(c)
            self._attrs["inputs"].append(a)
            self._attrs["inputs"].append(b)
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
        # problem_sizes_device: N * GemmCoord -> N * 3 * sizeof(int64_t) -> 32 * N
        # ptrA/B/C/D: N * 8 for each
        # lda/b/c/d: N * 8 for each
        # total: N * 8 * 4 + N * 8 * 4 + N * 8 * 4
        # total: 3 * 32 * N
        args_size = 96 * self._attrs["groups"]
        self._attrs["unique_workspace"] = args_size
        return ret

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        """Generate profiler for the op

        Parameters
        ----------
        workdir : str, optional
            [description], by default None
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy, used to filter generated profiles at compile time.
            See also: :func:`~aitemplate.compiler.transform.profile.profile`
        """
        target = Target.current()
        # init candidate ops
        func_key = "{target}.{op}.config".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        func(self._attrs)
        # init exec path
        self._extract_exec_path(dynamic_profiling_strategy)
        # init compile-time filter
        workloads = list(self._attrs["exec_path"].keys())
        ab_alignments = sorted({self._get_ab_alignment(wkl) for wkl in workloads})
        assert 1 == len(
            ab_alignments
        ), f"ab_alignments should be the same among all workloads, got {ab_alignments=}"
        func_key = "{target}.{op}.filter".format(
            target=target.name(), op=self._attrs["op"]
        )
        filter_func = registry.get(func_key)
        # run compile-time filter
        new_op_instance = OrderedDict(
            (k, v)
            for k, v in self._attrs["op_instance"].items()
            if filter_func(k, self._attrs, ab_alignments[0])
        )
        _LOGGER.debug(
            f"Filtered profiler kernels for {self._attrs['op']}: reduced the "
            f"number of generated kernels from {len(self._attrs['op_instance'])} "
            f"to {len(new_op_instance)}",
        )
        _LOGGER.debug(
            f"Group_gemm profiler valid configs: {sorted(new_op_instance.keys())}",
        )
        self._attrs["op_instance"] = new_op_instance
        build_profiler = super()._should_build_profiler(workloads, new_op_instance)
        if build_profiler:
            func_key = "{target}.{op}.gen_profiler".format(
                target=target.name(), op=self._attrs["op"]
            )
            func = registry.get(func_key)
            profiler_filename = self._get_profiler_filename()
            _LOGGER.info(f"generating {profiler_filename=}")
            return func(
                self._attrs, workdir, profiler_filename, self.shape_eval_template
            )

    def gen_function(self) -> str:
        """Generate function for the op

        Returns
        -------
        str
            C++ source code of the function
        """
        target = Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            self.exec_cond_template,
            self.shape_eval_template,
        )
