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
GEMM Specialization: (A.permute(0, 2, 1)[col] @ B[row])
Note: This op's output is a ColMajor
"""

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common
from aitemplate.compiler.ops.gemm_universal.bmm import bmm

# pylint: disable=C0103, W0223, W0221, W0613


class perm021fc_crc(bmm):
    """GEMM Specialization: (A.permute(0, 2, 1) @ B)

    This one is used when n/m gives you better alignment than m/k.
    Note: This op's output is a ColMajor

    This op is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(K, N).cuda().half()

        XT = X_pt.permute(0, 2, 1)
        XT = torch.reshape(XT, (-1, K))
        WT = W_pt.transpose(0, 1).contiguous()
        Y_pt = torch.nn.functional.linear(XT, WT)
        Y_pt = torch.reshape(Y_pt, (B, M, N)).contiguous()
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "perm021fc_crc"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(m, n, self._attrs["inputs"][0].dtype())

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        a_shapes = a._attrs["shape"]
        b_shapes = b._attrs["shape"]

        batch_size_a = a_shapes[0]
        batch_size_b = b_shapes[0]
        if (
            batch_size_a != batch_size_b
            and batch_size_a != IntImm(1)
            and batch_size_b != IntImm(1)
        ):
            raise RuntimeError(
                "bmm operand A and B should have same batch_size, or batch_size = 1! "
                "Current shape A: {} shape B: {} .".format(a_shapes, b_shapes)
            )
        batch_size = batch_size_b if batch_size_a == IntImm(1) else batch_size_a

        return [batch_size, b_shapes[2], a_shapes[2]]

    def _extract_dims(self, for_profiling=False):
        # (B, K, N) * (B, K, M) = (B, M, N)
        return {
            "B": [common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=0)],
            "M": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=2),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=2),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=2),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=1),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
            ],
        }

    def _invert_exec_key(self, key):
        return common.gemm_inverse_key_func(key)

    def _gen_profile_cmd(self, profiler_prefix, cfg, exec_key):
        def fbuild_cmd(exec_key):
            B, M, N, K = self._invert_exec_key(exec_key)
            cmd = []
            cmd.append(B)  # m
            cmd.append(M)  # m
            cmd.append(N)  # n
            cmd.append(K)  # k
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)

    def _align_ab(self, a: Tensor, b: Tensor):
        # b: [b, k, m]
        # a: [1, k, n]
        # TODO(xxx): Not implemented, need to pad m, n to 8
        return a, b
