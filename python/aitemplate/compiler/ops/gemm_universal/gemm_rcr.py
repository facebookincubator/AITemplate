# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""

from ...base import IntImm, Tensor
from . import gemm_common as common

# pylint: disable=C0103,W0223,W0221,W0613


class gemm_rcr(common.gemm):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "gemm_rcr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k)

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
