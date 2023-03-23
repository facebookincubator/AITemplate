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

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common
from aitemplate.compiler.ops.gemm_universal.bmm import bmm


class bmm_xxx(bmm):
    """Batch GEMM specialization"""

    def __init__(self, a_layout, b_layout, c_layout):
        super().__init__()
        self._attrs["op"] = f"bmm_{a_layout}{b_layout}{c_layout}"
        self.a_layout = a_layout
        self.b_layout = b_layout
        self.c_layout = c_layout

        self.a_is_column_major = int(self.a_layout == "c")
        self.b_is_column_major = int(self.b_layout == "c")
        self.c_is_column_major = int(self.c_layout == "c")

        def cal_align_ab(m, n, k):
            return common.default_align_ab(
                self._get_a_leading_dim(m, k),
                self._get_b_leading_dim(n, k),
                self._attrs["inputs"][0].dtype(),
            )

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        batch_size = self._get_batch_size(a, b)
        m = a.shape()[self._get_m_idx_in_a(a.shape())]
        n = b.shape()[self._get_n_idx_in_b(b.shape())]
        return [batch_size, *self._get_output_shape(m, n)]

    def _extract_dims(self, for_profiling=False):
        # C = A * B
        # A shape is (B, M, K) for row-major layout and (B, K, M) for column-major layout
        # B shape is (B, K, N) for row-major layout and (B, N, K) for column-major layout
        # C shape is (B, M, N) for row-major layout and (B, N, M) for column-major layout
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
                    common.Source.INPUT,
                    tensor_idx=0,
                    dim_idx=self._get_m_idx_in_a(a_shapes),
                ),
                common.DimInfo(
                    common.Source.OUTPUT,
                    tensor_idx=0,
                    dim_idx=self._get_m_idx_in_c(),
                ),
            ],
            "N": [
                common.DimInfo(
                    common.Source.INPUT,
                    tensor_idx=1,
                    dim_idx=self._get_n_idx_in_b(b_shapes),
                ),
                common.DimInfo(
                    common.Source.OUTPUT, tensor_idx=0, dim_idx=self._get_n_idx_in_c()
                ),
            ],
            "K": [
                common.DimInfo(
                    common.Source.INPUT,
                    tensor_idx=0,
                    dim_idx=self._get_k_idx_in_a(a_shapes),
                ),
                common.DimInfo(
                    common.Source.INPUT,
                    tensor_idx=1,
                    dim_idx=self._get_k_idx_in_b(b_shapes),
                ),
            ],
        }

        return dim_info_dict

    def _get_a_leading_dim(self, m, k):
        return [k, m][self.a_is_column_major]

    def _get_b_leading_dim(self, n, k):
        return [n, k][self.b_is_column_major]

    def _get_m_idx_in_a(self, a_shapes):
        return len(a_shapes) - 2 + self.a_is_column_major

    def _get_m_idx_in_c(self):
        return 1 + self.c_is_column_major

    def _get_n_idx_in_b(self, b_shapes):
        return len(b_shapes) - 1 - self.b_is_column_major

    def _get_n_idx_in_c(self):
        return 2 - self.c_is_column_major

    def _get_k_idx_in_a(self, a_shapes):
        return len(a_shapes) - 1 - self.a_is_column_major

    def _get_k_idx_in_b(self, b_shapes):
        return len(b_shapes) - 2 + self.b_is_column_major

    def _get_output_shape(self, m, n):
        if self.c_is_column_major:
            return [n, m]
        return [m, n]

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


class bmm_ccr(bmm_xxx):
    """Batch GEMM specialization for A[ColMajor], B[ColMajor], C[RowMajor].

    This operator is equivalent to following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        Y_pt = torch.bmm(XT, W_pt.transpose(2, 1))
    """

    def __init__(self):
        super().__init__("c", "c", "r")


class bmm_rrr(bmm_xxx):
    """Batch GEMM specialization for A[RowMajor], B[RowMajor], C[RowMajor]

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()

        Y_pt = torch.bmm(X_pt, W_pt)
    """

    def __init__(self):
        super().__init__("r", "r", "r")


class bmm_crr(bmm_xxx):
    """Batch GEMM specialization for A[ColMajor], B[RowMajor], C[RowMajor].

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        Y_pt = torch.bmm(XT, W_pt)

    """

    def __init__(self):
        super().__init__("c", "r", "r")


class bmm_rcr(bmm_xxx):
    """Batch GEMM specialization for A[RowMajor], B[ColMajor], C[RowMajor].

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        Y_pt = torch.bmm(XT, W_pt)

    """

    def __init__(self):
        super().__init__("r", "c", "r")


class bmm_ccc(bmm_xxx):
    """Batch GEMM specialization for A[ColMajor], B[ColMajor], C[ColMajor].

    This operator is equivalent to following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        YT = torch.bmm(XT, W_pt.transpose(2, 1))
        Y_pt = torch.transpose(YT, 2, 1)
    """

    def __init__(self):
        super().__init__("c", "c", "c")


class bmm_rrc(bmm_xxx):
    """Batch GEMM specialization for A[RowMajor], B[RowMajor], C[ColMajor]

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()

        YT = torch.bmm(X_pt, W_pt)
        Y_pt = torch.transpose(YT, 2, 1)
    """

    def __init__(self):
        super().__init__("r", "r", "c")


class bmm_crc(bmm_xxx):
    """Batch GEMM specialization for A[ColMajor], B[RowMajor], C[ColMajor].

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        YT = torch.bmm(XT, W_pt)
        Y_pt = torch.transpose(YT, 2, 1)

    """

    def __init__(self):
        super().__init__("c", "r", "c")


class bmm_rcc(bmm_xxx):
    """Batch GEMM specialization for A[RowMajor], B[ColMajor], C[ColMajor].

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        YT = torch.bmm(XT, W_pt)
        Y_pt = torch.transpose(YT, 2, 1)

    """

    def __init__(self):
        super().__init__("r", "c", "c")
