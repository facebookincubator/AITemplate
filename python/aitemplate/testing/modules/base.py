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

import os
from enum import Enum

import torch

from aitemplate.compiler import compile_model
from aitemplate.testing import detect_target

ExecutionMode = Enum("ExecutionMode", ["RUN", "PROFILE", "BENCHMARK"])


class AITModule(torch.nn.Module):
    def __init__(self, workdir, target, module_name, mode):
        super(AITModule, self).__init__()

        self.engine = None
        self.workdir = workdir
        self.module_name = module_name
        self.mode = mode
        self.target = detect_target() if target is None else target

    def _get_profile_report_filename(self):
        return os.path.join(
            self.workdir, self.module_name, f"profile-{self.module_name}.json"
        )

    def _maybe_compile(self, *pt_inputs):
        if not self.engine:
            graph = self._create_graph(*pt_inputs)
            self.engine = compile_model(
                graph, self.target, self.workdir, self.module_name
            )

    def _run(self, inputs, outputs):
        if self.mode == ExecutionMode.RUN:
            self.engine.run_with_tensors(inputs, outputs)
        elif self.mode == ExecutionMode.PROFILE:
            self.engine.profile_with_tensors(
                inputs,
                outputs,
                num_iters=1000,
                filename=self._get_profile_report_filename(),
            )
        elif self.mode == ExecutionMode.BENCHMARK:
            self.engine.benchmark_with_tensors(inputs, outputs)
