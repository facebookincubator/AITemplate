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
Batch GEMM specialization for A[RowMajor], B[ColMajor], C[RowMajor].
"""

from ...base import Tensor
from . import gemm_common as common
from .bmm import bmm

# pylint: disable=C0103, W0223, W0221, W0613


class bmm_rcr(bmm):
    """Batch GEMM specialization for A[RowMajor], B[ColMajor], C[RowMajor].

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        Y_pt = torch.bmm(XT, W_pt)

    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "bmm_rcr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k, self._attrs["inputs"][0].dtype())

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        batch_size = self._get_batch_size(a, b)
        return [batch_size, a.shape()[-2], b.shape()[-2]]

    def _extract_dims(self, for_profiling=False):
        # (B, M, K) * (B, N, K) = (B, M, N)
        a_shapes = common.extract_shape_from_accessor(
            self._attrs, common.Source.INPUT, 0
        )
        b_shapes = common.extract_shape_from_accessor(
            self._attrs, common.Source.INPUT, 1
        )
        output_shapes = common.extract_shape_from_accessor(
            self._attrs, common.Source.OUTPUT, 0
        )

        B_dim = common.create_input_batch_diminfo(
            [a_shapes, b_shapes], [0, 0], output_shapes[0]
        )
        B_dim.append(common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=0))

        dim_info_dict = {
            "B": B_dim,
            "M": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=0, dim_idx=len(a_shapes) - 2
                ),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1),
            ],
            "N": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=1, dim_idx=len(b_shapes) - 2
                ),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=2),
            ],
            "K": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=0, dim_idx=len(a_shapes) - 1
                ),
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=1, dim_idx=len(b_shapes) - 1
                ),
            ],
        }

        return dim_info_dict

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
