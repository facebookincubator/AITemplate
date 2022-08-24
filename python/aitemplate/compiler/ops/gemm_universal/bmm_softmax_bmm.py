# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""

from ...base import IntImm, Tensor
from ...tensor_accessor import TensorAccessor
from . import gemm_common as common
from .bmm import bmm

# pylint: disable=C0103, W0223, W0221, W0613


class bmm_softmax_bmm(bmm):
    """BatchGemm,
    A: row_major, B: column_major, C: row_major,
    A: [b, m, k], B: [b, n, k], C: [b, m, n]
    """

    def __init__(self, scale=1.0):
        """_summary_"""
        super().__init__()
        self._attrs["op"] = "bmm_softmax_bmm"
        self._attrs["scale"] = scale

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k)

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor, b1: Tensor):
        a_shapes = a._attrs["shape"]
        b_shapes = b._attrs["shape"]
        b1_shapes = b1._attrs["shape"]

        batch_size_a = a_shapes[0]
        batch_size_b = b_shapes[0]
        if batch_size_a != batch_size_b and batch_size_a != 1 and batch_size_b != 1:
            raise RuntimeError(
                "bmm_rcr operand A and B should have same batch_size, or batch_size = 1! "
                "Current shape A: {} shape B: {} .".format(a_shapes, b_shapes)
            )
        batch_size = batch_size_b if batch_size_a == IntImm(1) else batch_size_a
        assert (
            a_shapes[2] == b_shapes[2]
        ), f"bmm_rcr operand A and B should have the same K dim (dim2)! Current shape A: {a_shapes}, shape B: {b_shapes}"
        return [batch_size, a_shapes[1], b1_shapes[2]]

    def _extract_dims(self, for_profiling=False):
        # (B, M, K) * (B, N, K) = (B, M, N)
        # softmax on (B, M, N)
        # (B, M, N) * (B, N, O) = (B, M, O)
        return {
            # TODO: support BMM broadcast
            "B": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=0),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=0),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=0),
            ],
            "M": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=1),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=1),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=2),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=2),
            ],
            "O": [
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=2),
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
            B, M, N, K, C = self._invert_exec_key(exec_key)
            cmd = []
            cmd.append(B)  # m
            cmd.append(M)  # m
            cmd.append(N)  # n
            cmd.append(K)  # k
            cmd.append(C)  # o
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)

    def __call__(self, a: Tensor, b: Tensor, b1: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, b1]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, b1)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
