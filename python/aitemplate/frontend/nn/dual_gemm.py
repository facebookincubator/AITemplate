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
Frontend for attention module
"""
from aitemplate.compiler import ops
from aitemplate.frontend.nn.linear import Linear
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter

# pylint: disable=C0103


class DualGemm(Module):
    r"""DualGemm frontend"""

    def __init__(
        self,
        in_channels,
        out_channels,
        fast_gelu=True,
        dtype="float16",
    ):
        """Initialize dual gemm module, create a tensor for weights"""
        super().__init__()
        self.w1 = Parameter(shape=[out_channels, in_channels], dtype=dtype)
        self.w2 = Parameter(shape=[out_channels, in_channels], dtype=dtype)
        if fast_gelu:
            self.op = ops.dual_gemm_rcr_fast_gelu()
        else:
            self.op = ops.dual_gemm_rcr_silu()

    def forward(self, *args):
        """forward pass for calling attention op"""
        assert len(args) == 1
        x = args[0]
        return self.op(x, self.w1.tensor(), self.w2.tensor())


class T5DenseGatedGeluDense(Module):
    r"""T5DenseGatedGeluDense."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dtype="float16",
    ):
        super().__init__()
        self.wi_0_weight = Parameter(
            shape=[out_channels, in_channels],
            dtype=dtype,
        )
        self.wi_1_weight = Parameter(
            shape=[out_channels, in_channels],
            dtype=dtype,
        )
        self.wo = Linear(
            out_channels,
            in_channels,
            bias=False,
            dtype=dtype,
        )
        self.op = ops.dual_gemm_rcr_fast_gelu()

    def forward(self, *args):
        """forward pass for calling T5 block"""
        assert len(args) == 1
        x = args[0]
        hidden = self.op(x, self.wi_0_weight.tensor(), self.wi_1_weight.tensor())
        return self.wo(hidden)
