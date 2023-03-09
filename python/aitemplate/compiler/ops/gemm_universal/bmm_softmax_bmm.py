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
BMM_RCR + Softmax + BMM_RRR Specialization
"""

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_common as common
from aitemplate.compiler.ops.gemm_universal.bmm import bmm
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103, W0223, W0221, W0613


class bmm_softmax_bmm(bmm):
    """BMM_RCR + Softmax + BMM_RRR Specialization
    This fusion is commonly used in Attention family

    This op is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        Q = torch.randn(B, M, K).cuda().half()
        K = torch.randn(B, N, K).cuda().half()
        V = torch.randn(B, N, O).cuda().half()

        attn = torch.bmm(Q, K.transpose(1, 2)) * scale
        attn = torch.softmax(attn, dim=-1)
        score = torch.bmm(attn, V)

    Limitations:
    1. Output dim O should be smaller than 256.
    2. CUDA backend codegen is not implemented in this release.
    """

    def __init__(self, scale=1.0):
        """Constructor of BMM_RCR + Softmax + BMM_RRR op

        Parameters
        ----------
        scale : float, optional
            normalization factor, by default 1.0

        """
        super().__init__()
        self._attrs["op"] = "bmm_softmax_bmm"
        self._attrs["scale"] = scale

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k, self._attrs["inputs"][0].dtype())

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
        return common.gemm_inverse_key_func(key)

    def _gen_profile_cmd(self, profiler_prefix, cfg, exec_key):
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
        """Call the BMM_RCR + Softmax + BMM_RRR op

        Parameters
        ----------
        a : Tensor
            Tensor in shape of [B, M, K]
        b : Tensor
            Tensor in shape of [B, N, K]
        b1 : Tensor
            Tensor in shape of [B, N, O]

        Returns
        -------
        Tensor
            Tensors in shape of [B, M, O]
        """
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b, b1]
        self._attrs["input_accessors"] = [
            TensorAccessor(a),
            TensorAccessor(b),
            TensorAccessor(b1),
        ]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, b1)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output

    def _get_op_attributes(self):
        return {"scale": self._attrs["scale"]}
