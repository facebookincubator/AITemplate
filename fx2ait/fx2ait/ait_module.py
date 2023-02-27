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
from typing import List

import torch


class AITModule(torch.nn.Module):
    def __init__(
        self,
        engine=None,
        interp_result=None,
    ):
        super(AITModule, self).__init__()
        self.engine = engine
        self.interp_result = interp_result

    def forward(self, *inputs):
        python_inputs = []
        if self.interp_result:
            inputs = list(inputs)
            for name, inp in zip(self.interp_result.fx_input_names, inputs):
                if name in self.interp_result.input_names:
                    python_inputs.append(inp)
            assert len(python_inputs) == len(self.interp_result.input_names)
        else:
            python_inputs = inputs

        outputs = self.engine.forward(python_inputs)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def profile(
        self, inputs: List[torch.Tensor], filename: str, num_iters: int
    ) -> None:
        """
        Profile the AIT module and save the report to a file. The AITModule
        must be created with allow_scripting=False.
        inputs: sample inputs
        filename: report filename
        num_iters: number of iterations per op run
        """
        self.engine.profile(inputs, filename, num_iters)

    @staticmethod
    def create_ait_module_wrapper(engine, interp_result, trace_ait_module, *inputs):
        """
        Some use cases need to torch.jit.script a model with AITModules in
        it, but TorchScript does not support variadic inputs. We can get
        around this by scripting the AITModule with some sample inputs.
        This is turned in by passing allow_scripting=True.
        """
        mod = AITModule(engine, interp_result)
        return torch.jit.trace(mod, inputs) if trace_ait_module else mod
