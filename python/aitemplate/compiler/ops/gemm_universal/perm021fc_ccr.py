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
GEMM Specialization: A.permute(0, 2, 1)[col] @ B[col]
"""

from aitemplate.compiler.base import _create_host_zero_tensor, IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common
from aitemplate.compiler.ops.gemm_universal.bmm import bmm
from aitemplate.compiler.ops.tensor import concatenate
from aitemplate.utils import alignment

# pylint: disable=C0103, W0223, W0221, W0613


class perm021fc_ccr(bmm):
    """GEMM Specialization: A.permute(0, 2, 1) @ B

    This op is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python
        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()

        XT = X_pt.permute(0, 2, 1)
        XT = torch.reshape(XT, (-1, K))
        Y_pt = torch.nn.functional.linear(XT, W_pt)
        Y_pt = torch.reshape(Y_pt, (B, M, N))
    """

    def __init__(self):
        """Constructor for perm021fc_ccr"""
        super().__init__()
        self._attrs["op"] = "perm021fc_ccr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(m, k, self._attrs["inputs"][0].dtype())

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

        return [batch_size, a_shapes[2], b_shapes[1]]

    def _extract_dims(self, for_profiling=False):
        # (B, K, M) * (B, N, K) = (B, M, N)
        return {
            "B": [common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=0)],
            "M": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=2),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=2),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=1),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=2),
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
        # a: [b, k, m]
        # b: [1, n, k]
        a_shape = a._attrs["shape"]
        b_shape = b._attrs["shape"]
        ak = a_shape[1]
        bk = b_shape[2]
        if ak != bk:
            raise RuntimeError(
                f"A/B shape mismatch, ak: {ak}, bk: {bk}, "
                f"a_shape: {a_shape}, b_shape: {b_shape}"
            )
        if not isinstance(bk, IntImm):
            raise RuntimeError(
                "Last dim K must be static! Current shape: {}".format(b_shape)
            )
        k = ak._attrs["values"][0]

        if not alignment.valid_alignment(k % 2, a.dtype()):
            pad_k = int((k // 8 + 1) * 8)

            pad_a = _create_host_zero_tensor(
                shape=[
                    a_shape[0],
                    IntImm(pad_k - k),
                    a_shape[2],
                ],
                dtype=a.dtype(),
            )
            pad_b = _create_host_zero_tensor(
                shape=[
                    b_shape[0],
                    b_shape[1],
                    IntImm(pad_k - k),
                ],
                dtype=b.dtype(),
            )
            cat_a = concatenate()
            cat_b = concatenate()
            a = cat_a([a, pad_a], dim=1)
            b = cat_b([b, pad_b], dim=2)
        return a, b
