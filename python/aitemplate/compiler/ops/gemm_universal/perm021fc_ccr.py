# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[b, m, n] = bmm([b, k, m], [1, n, k])
in torch it is
...
# _3306 = _3305.permute(0, 2, 1)  # Transpose
# _3307 = _3306  # torch.reshape(_3306, (-1, 745))  # Reshape
# _3308 = torch.nn.functional.linear(_3307, self._1184, bias=self._1185)  # FC
"""

from ...base import IntImm, Tensor
from ..tensor import concatenate
from . import gemm_common as common
from .bmm import bmm

# pylint: disable=C0103, W0223, W0221, W0613


class perm021fc_ccr(bmm):
    """_summary_

    Parameters
    ----------
    common : _type_
        _description_
    """

    def __init__(self):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "perm021fc_ccr"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(m, k)

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

        if k % 2 != 0:
            pad_k = int((k // 8 + 1) * 8)

            pad_a = Tensor(
                shape=[
                    a_shape[0],
                    IntImm(pad_k - k),
                    a_shape[2],
                ],
                dtype=a.dtype(),
            )
            pad_b = Tensor(
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
