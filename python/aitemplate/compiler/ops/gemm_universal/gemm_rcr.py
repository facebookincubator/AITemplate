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
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
"""

from aitemplate.backend import registry, target
from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr(common.gemm):
    """GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(N, K).cuda().half()

        y = torch.nn.functional.linear(A, B)
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gemm_rcr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k, self._attrs["inputs"][0].dtype())

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        return a._attrs["shape"][:-1] + [b._attrs["shape"][0]]

    def _extract_dims(self, for_profiling=False):
        # (M, K) * (N, K) = (M, N)

        # profiling always uses 2d * 2d.
        A_len = (
            2
            if for_profiling
            else len(self._attrs["input_accessors"][0].original_shapes)
        )
        return {
            "M": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=0, dim_idx=list(range(A_len - 1))
                ),
                common.DimInfo(
                    common.Source.OUTPUT, tensor_idx=0, dim_idx=list(range(A_len - 1))
                ),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=0),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=A_len - 1),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=A_len - 1),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
            ],
        }

    def _invert_exec_key(self, key):
        return common.gemm_inverse_key_func(key)

    def _gen_profile_cmd(self, profiler_prefix, cfg, exec_key):
        def fbuild_cmd(exec_key):
            M, N, K = self._invert_exec_key(exec_key)
            cmd = []
            cmd.append(M)  # m
            cmd.append(N)  # n
            cmd.append(K)  # k
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)

    def _align_ab(self, a: Tensor, b: Tensor):
        a_shape = a._attrs["shape"]
        b_shape = b._attrs["shape"]
        if a_shape[-1] != b_shape[-1]:
            raise RuntimeError(
                "A/B shape mismatch! A: {}, B: {}".format(a_shape, b_shape)
            )
        if not isinstance(a_shape[-1], IntImm):
            raise RuntimeError("K must be static! k: {}".format(a_shape[-1]))

        return a, b

    def _use_cutedsl(self) -> bool:
        """Check if CuTeDSL backend is enabled for this op."""
        current_target = target.Target.current()
        return current_target._kwargs.get("use_cutedsl_gemm", False)

    def _backend_suffix(self) -> str:
        """Return registry key suffix for the active backend."""
        return "_cutedsl" if self._use_cutedsl() else ""

    def gen_function(self) -> str:
        """Generate function code, dispatching to CuTeDSL if enabled."""
        current_target = target.Target.current()
        suffix = self._backend_suffix()
        self._attrs["backend_suffix"] = suffix
        func_key = "{target}.{op}.gen_function{suffix}".format(
            target=current_target.name(),
            op=self._attrs["op"],
            suffix=suffix,
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            self.exec_cond_template,
            self._extract_dims(),
        )

    def gen_function_decl(self, func_attrs=None) -> str:
        """Generate function declaration, dispatching to CuTeDSL if enabled."""
        current_target = target.Target.current()
        suffix = self._backend_suffix()
        func_key = "{target}.{op}.func_decl{suffix}".format(
            target=current_target.name(),
            op=self._attrs["op"],
            suffix=suffix,
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_call(self, func_attrs=None, indent="  ") -> str:
        """Generate function call, dispatching to CuTeDSL if enabled."""
        current_target = target.Target.current()
        suffix = self._backend_suffix()
        func_key = "{target}.{op}.func_call{suffix}".format(
            target=current_target.name(),
            op=self._attrs["op"],
            suffix=suffix,
        )
        func = registry.get(func_key)
        return func(self._attrs, indent=indent)
