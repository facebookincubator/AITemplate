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
GEMM Specialization: (A.permute(0, 2, 1)[col] @ B[row] + Bias)
"""

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import perm021fc_crc
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103, W0223, W0221


class perm021fc_crc_bias(perm021fc_crc):
    """GEMM Specialization: (A.permute(0, 2, 1) @ B + Bias)

    This one is used when n/m gives you better alignment than m/k.

    This op is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(K, N).cuda().half()
        B_pt = torch.randn(N).cuda().half()

        XT = X_pt.permute(0, 2, 1)
        XT = torch.reshape(XT, (-1, K))
        WT = W_pt.transpose(0, 1).contiguous()
        Y_pt = torch.nn.functional.linear(XT, WT, bias=B_pt)
        Y_pt = torch.reshape(Y_pt, (B, M, N)).contiguous()
    """

    def __init__(self):
        """Constractor for perm021fc_crc_bias"""
        super().__init__()
        self._attrs["op"] = "perm021fc_crc_bias"

    def _infer_shapes(self, a: Tensor, b: Tensor, bias: Tensor):
        bias_shape = bias._attrs["shape"]
        if len(bias_shape) != 1:
            raise RuntimeError("Bias should be 1D vector ")
        bias_dim = bias_shape[0]
        if not isinstance(bias_dim, IntImm):
            raise RuntimeError("Bias should be fixed 1D vector")
        outshape = super()._infer_shapes(a, b)
        if outshape[2] != bias_dim:
            raise RuntimeError("GEMM/Bias shape doesn't match")
        return outshape

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        self._attrs["inputs"] = [a, b, bias]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b, bias)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=a._attrs["dtype"])
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["outputs"]
        ]
        return output
