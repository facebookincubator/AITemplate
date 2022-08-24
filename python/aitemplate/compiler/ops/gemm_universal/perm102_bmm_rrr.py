# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row))
in torch it is
# _2905_2929 = _2904.view(B, 25, -1).permute(1, 0, 2)
# _2930_2954 = torch.baddbmm(
#      self._1085_1133, _2905_2929, self._1084_1132) # baddbmm(bias, X, W)
"""

from ...base import IntImm, Tensor
from . import gemm_common as common
from .bmm import bmm

# pylint: disable=C0103, W0223, W0221, W0613


class perm102_bmm_rrr(bmm):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "perm102_bmm_rrr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, n)

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        a_shapes = a._attrs["shape"]
        b_shapes = b._attrs["shape"]

        batch_size_a = a_shapes[1]
        batch_size_b = b_shapes[0]
        if batch_size_a != batch_size_b and batch_size_a != 1 and batch_size_b != 1:
            raise RuntimeError(
                "bmm operand A and B should have same batch_size, or batch_size = 1! "
                "Current shape A: {} shape B: {} .".format(a_shapes, b_shapes)
            )
        batch_size = batch_size_b if batch_size_a == IntImm(1) else batch_size_a

        return [a_shapes[0], batch_size, b_shapes[2]]

    def _extract_dims(self, for_profiling=False):
        # (M, B, K) * (B, K, N) = (M, B, N)
        return {
            "B": [common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1)],
            "M": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=0),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=0),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=2),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=2),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=2),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
            ],
        }

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
            cmd.append(B)  # b
            cmd.append(M)  # m
            cmd.append(N)  # n
            cmd.append(K)  # k
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)
