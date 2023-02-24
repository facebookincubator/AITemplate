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

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntVar
from aitemplate.frontend import Tensor
from aitemplate.testing.modules.base import AITModule, ExecutionMode
from aitemplate.utils.torch_utils import torch_dtype_to_string


class SoftmaxAIT(AITModule):
    def __init__(
        self,
        dim,
        batch_sizes=(1,),
        module_name="softmax",
        workdir="./tmp",
        target=None,
        mode=ExecutionMode.RUN,
    ):
        super(SoftmaxAIT, self).__init__(workdir, target, module_name, mode)
        self.dim = dim
        self.batch_sizes = batch_sizes
        self.op = ops.softmax()

    def _create_graph(self, x_pt):
        X = Tensor(
            shape=[
                IntVar(name="input_batch", values=list(self.batch_sizes)),
                *x_pt.shape[1:],
            ],
            dtype=torch_dtype_to_string(x_pt.dtype),
            name="X",
            is_input=True,
        )
        Y = self.op(X, self.dim)
        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True
        return Y

    def forward(self, x_pt, out=None):
        self._maybe_compile(x_pt)
        if out is None:
            out = torch.empty_like(x_pt)
        self._run({"X": x_pt}, {"Y": out})
        return out
