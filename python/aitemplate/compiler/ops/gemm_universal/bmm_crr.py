# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When used for `linear`, need to set A->Data, B->Weight
"""

from ...base import Tensor
from . import gemm_common as common
from .bmm import bmm

# pylint: disable=C0103, W0223, W0221, W0613


class bmm_crr(bmm):
    """BatchGemm,
    A: column_major, B: row_major, C: row_major,
    A: [b, k, m], B: [b, k, n], C: [b, m, n]
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "bmm_crr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(m, n)

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        batch_size = self._get_batch_size(a, b)
        return [batch_size, a.shape()[-1], b.shape()[-1]]

    def _extract_dims(self, for_profiling=False):
        # (B, K, M) * (B, K, N) = (B, M, N)
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
                    common.Source.INPUT, tensor_idx=0, dim_idx=len(a_shapes) - 1
                ),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1),
            ],
            "N": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=1, dim_idx=len(b_shapes) - 1
                ),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=2),
            ],
            "K": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=0, dim_idx=len(a_shapes) - 2
                ),
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=1, dim_idx=len(b_shapes) - 2
                ),
            ],
        }

        return dim_info_dict

    def _invert_exec_key(self, key):
        """_summary_

        Parameters
        ----------
        key : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return common.gemm_inverse_key_func(key)

    def _gen_profile_cmd(self, profiler_prefix, cfg, exec_key):
        """_summary_

        Parameters
        ----------
        profiler_prefix : _type_
            _description_
        cfg : _type_
            _description_
        exec_key : _type_
            _description_
        """

        def fbuild_cmd(exec_key):
            B, M, N, K = self._invert_exec_key(exec_key)
            cmd = []
            cmd.append(B)  # m
            cmd.append(M)  # m
            cmd.append(N)  # n
            cmd.append(K)  # k
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)
