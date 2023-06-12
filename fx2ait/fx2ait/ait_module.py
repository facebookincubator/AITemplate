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

ARG_SPLITTER_KEYWORD = "a1T_ARg_SpliTTERKeyword"


class AITModule(torch.nn.Module):
    def __init__(
        self,
        engine=None,
        interp_result=None,
    ):
        super(AITModule, self).__init__()
        self.engine = engine

        self.interp_result = interp_result
        self.ait_arg_names = interp_result.input_names if interp_result else None
        self.fx_arg_names = interp_result.fx_input_names if interp_result else None

    def forward(self, *args, **kwargs):
        ait_args = []
        if self.interp_result:
            offset = 0
            for idx, fx_arg_name in enumerate(self.fx_arg_names):
                arg_name, *arg_idx = fx_arg_name.split(ARG_SPLITTER_KEYWORD)
                arg_idx = int(arg_idx[0]) if arg_idx else -1
                # Offset for List[List[Tensor]]
                offset += 1 if arg_idx > 0 else 0
                if fx_arg_name in self.ait_arg_names:
                    # Locate input from args.
                    if idx - offset < len(args):
                        arg_ref = args[idx - offset]
                    # Locate input from kwargs.
                    elif arg_name in kwargs:
                        arg_ref = kwargs[arg_name]
                    else:
                        raise RuntimeError(f"Required input {fx_arg_name} not found")
                    ait_args.append(arg_ref[arg_idx] if arg_idx > -1 else arg_ref)

            assert len(ait_args) == len(self.ait_arg_names)
        else:
            # Flatten args and kwargs from List[Tensor or List[Tensor]] to List[Tensor]
            all_args = list(args) + list(kwargs.values())
            for arg in all_args:
                ait_args.extend(arg if isinstance(arg, list) else [arg])

        outputs = self.engine.forward(ait_args)
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
        # sanity test before tracing
        mod(*inputs)
        return torch.jit.trace(mod, inputs) if trace_ait_module else mod
