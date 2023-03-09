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
LayerNorm module.
"""
from aitemplate.compiler import ops
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter

# pylint: disable=C0103


class LayerNorm(Module):
    """LayerNorm nn module"""

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        dtype="float16",
        **kwargs,
    ):
        """Standalone layernorm op.
        Applies Layer Normalization over a mini-batch of inputs as described in the
        paper Layer Normalization. The mean and standard-deviation are calculated
        over the last D dimensions, where D is the dimension of normalized_shape.
        Input shape: [M0, M1, ..., Mp, N1, N2, ..., ND]
        Normalized_shape: [N1, N2, ..., ND]
        Gamma/Beta, if not None, have the same shape as normalized_shape.
        """
        super().__init__()
        self.eps = eps
        self.dim = (
            normalized_shape
            if isinstance(normalized_shape, (tuple, list))
            else (normalized_shape,)
        )
        self.weight = Parameter(shape=self.dim, dtype=dtype)
        self.bias = Parameter(shape=self.dim, dtype=dtype)
        self.op = ops.layernorm()

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        y = self.op(x, self.weight.tensor(), self.bias.tensor(), self.dim, self.eps)
        return y
