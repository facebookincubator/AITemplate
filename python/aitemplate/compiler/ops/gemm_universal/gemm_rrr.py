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
GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rrr(common.gemm):
    """GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(K, N).cuda().half()

        y = torch.nn.functional.linear(A, B.t())
    """

    def __init__(self):
        """Constructor of gemm_rrr"""
        super().__init__()
        self._attrs["op"] = "gemm_rrr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, n, self._attrs["inputs"][0].dtype())

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        return a._attrs["shape"][:-1] + [b._attrs["shape"][1]]

    def _extract_dims(self, for_profiling=False):
        # (M, K) * (K, N) = (M, N)

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
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=A_len - 1),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=A_len - 1),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=0),
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
        if a_shape[-1] != b_shape[0]:
            raise RuntimeError(
                "A/B shape mismatch! A: {}, B: {}".format(a_shape, b_shape)
            )
        if not isinstance(a_shape[-1], IntImm):
            raise RuntimeError("K must be static! k: {}".format(a_shape[-1]))
        return a, b
