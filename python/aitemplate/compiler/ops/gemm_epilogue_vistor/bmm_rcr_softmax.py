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
Operator definition for bmm_rcr_softmax.
"""

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common
from aitemplate.compiler.ops.gemm_universal.bmm import bmm
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103, W0223, W0221, W0613


class bmm_rcr_softmax(bmm):
    """BatchGemm with softmax,
    A: row_major, B: column_major, C: row_major,
    A: [b, m, k], B: [b, n, k], C: [b, m, n]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "bmm_rcr_softmax"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k, self._attrs["inputs"][0].dtype())

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        a_shapes = a._attrs["shape"]
        b_shapes = b._attrs["shape"]

        batch_size_a = a_shapes[0]
        batch_size_b = b_shapes[0]
        if batch_size_a != batch_size_b and batch_size_a != 1 and batch_size_b != 1:
            raise RuntimeError(
                "bmm operand A and B should have same batch_size, or batch_size = 1! "
                "Current shape A: {} shape B: {} .".format(a_shapes, b_shapes)
            )
        batch_size = batch_size_b if batch_size_a == IntImm(1) else batch_size_a
        return [batch_size, a_shapes[1], b_shapes[1]]

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        """Performs sanity checks, offline shape inference and returns an output tensor."""

        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b]

        self._sanity_check(a, b)

        output_shape = self._infer_shapes(a, b)
        self._extract_epilogue_alignment(output_shape)

        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]

        self._set_depth()

        output = Tensor(output_shape, src_ops={self}, dtype=a._attrs["dtype"])
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output

    def _extract_dims(self, for_profiling=False):
        # (B, M, K) * (B, N, K) = (B, M, N)
        return {
            "B": [common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=0)],
            "M": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=1),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=2),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=2),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=2),
            ],
        }

    def _invert_exec_key(self, key):
        """extract tensor shape info from key"""
        return common.gemm_inverse_key_func(key)

    def _gen_profile_cmd(self, profiler_prefix, cfg, exec_key):
        """get command line for profiling"""

        def fbuild_cmd(exec_key):
            B, M, N, K = self._invert_exec_key(exec_key)
            cmd = []
            cmd.append(B)  # b
            cmd.append(M)  # m
            cmd.append(N)  # n
            cmd.append(K)  # k
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)
