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
Linear module.
"""
from aitemplate.testing import detect_target

from ...compiler import ops
from .module import Module
from .parameter import Parameter

# pylint: disable=C0103

USE_CUDA = detect_target().name() == "cuda"


class Linear(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        specialization=None,
        dtype="float16",
        **kwargs,
    ):
        super().__init__()
        self.weight = Parameter(shape=[out_channels, in_channels], dtype=dtype)
        op_name = "gemm_rcr_bias" if bias else "gemm_rcr"
        if specialization is not None:
            op_name += "_" + specialization
        if bias:
            self.bias = Parameter(shape=[out_channels], dtype=dtype)
        op_func = getattr(ops, op_name)
        self._op_name = op_name
        self.op = op_func(**kwargs)
        self.use_bias = bias
        self.in_channels = in_channels

    def forward(self, *args):
        assert len(args) >= 1
        x = args[0]
        if not USE_CUDA:
            shape = x._attrs["shape"]
            x = x if len(shape) == 2 else ops.reshape()(x, [-1, self.in_channels])
        if len(args) == 2:
            if self.use_bias:
                inputs = [x, self.weight.tensor(), self.bias.tensor(), args[1]]
            else:
                inputs = [x, self.weight.tensor(), args[1]]
            output = self.op(*inputs)
            return output
        output = (
            self.op(x, self.weight.tensor(), bias=self.bias.tensor())
            if self.use_bias
            else self.op(x, self.weight.tensor())
        )
        return output
