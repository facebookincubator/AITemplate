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
GEMM Specialization: (A.permute(0, 2, 1)[col] @ B[col] + Bias).permute(0, 2, 1)
"""
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common import reshape
from aitemplate.compiler.ops.gemm_universal.perm021fc_ccr_bias import perm021fc_ccr_bias
from aitemplate.compiler.tensor_accessor import TensorAccessor

# pylint: disable=C0103,W0223,W0221,W0613


class perm021fc_ccr_bias_permute(perm021fc_ccr_bias):
    """
    GEMM Specialization: (A.permute(0, 2, 1) @ B + Bias).permute(0, 2, 1)

    Note: This fusion may be slower than the non-fused version due to NVCC
    is not able to optimize the fused version.

    This op is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python
        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(N, K).cuda().half()
        Bias_pt = torch.randn(N).cuda().half()

        XT = X_pt.permute(0, 2, 1)
        XT = torch.reshape(XT, (-1, K))
        Y_pt = torch.nn.functional.linear(XT, W_pt, bias=B_pt)
        Y_pt = torch.reshape(Y_pt, (B, M, N))
        Y_pt = Y_pt.permute(0, 2, 1)
    """

    def __init__(self, layout="021"):
        """Constructor for perm021fc_ccr_bias_permute"""
        super().__init__()
        self._attrs["op"] = "perm021fc_ccr_bias_permute"
        self._attrs["shape"] = [0]  # this is a hack
        self._attrs["layout"] = "Permute3DBMM_{}".format(layout)

    def __call__(self, a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
        a, b = self._align_ab(a, b)
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
        self._attrs["output_accessors"] = [TensorAccessor(output)]

        if self._attrs["layout"] == "Permute3DBMM_021":
            b, m, n = output_shape
            output_shape = [b, n, m]
            self._attrs["epilogue_alignment"] = 1
            return reshape()(output, output_shape)
        else:
            raise NotImplementedError(
                "{} is not implemented!".format(self._attrs["layout"])
            )

    def _get_op_attributes(self):
        return {"layout": self._attrs["layout"].split("_")[-1]}
