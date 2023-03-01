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
import logging
from typing import Any, Callable, List, Tuple

import torch
import torch.fx.passes.net_min_base as net_min_base

from fx2ait.ait_module import AITModule
from fx2ait.fx2ait import AITInterpreter, TensorSpec
from torch.fx.passes.tools_common import Tensors

_LOGGER: logging.Logger = logging.getLogger(__name__)


def lower_mod_default(
    mod: torch.fx.GraphModule,
    inputs: List[TensorSpec],
    workdir: str,
    name: str,
    dll_name: str,
) -> AITModule:
    interp = AITInterpreter(mod, inputs, workdir, name, dll_name)
    interpreter_result = interp.run()
    res_mod = AITModule(
        torch.classes.fb.AITModel(
            interpreter_result.engine.lib_path,
            interpreter_result.input_names,
            interpreter_result.output_names,
            torch.float16,
            torch.float16,
            1,  # num_runtimes
        ),
        interpreter_result,
    )
    return res_mod


class AITMinizerSetting(net_min_base._MinimizerSettingBase):
    def __init__(self):
        super().__init__()


class AITMinimizer(net_min_base._MinimizerBase):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tensors,
        compare_fn: Callable[[Any, Any, Any], Tuple[float, bool]] = lambda a, b, c: (
            torch.dist(a, b),
            torch.allclose(a, b),
        ),
        settings: AITMinizerSetting = AITMinizerSetting(),
        lower_fn: Callable[
            [torch.fx.GraphModule, Tensors, str, str, str], AITModule
        ] = lower_mod_default,
        workdir: str = "./tmp/AITMinimizer",
        name: str = "minimize_module",
    ):
        self.lower_fn = lower_fn
        self.workdir = workdir
        self.name = name
        self.curr_iter = 0  # We use this counter to prevent duplicate .so naming
        super().__init__(module, sample_input, compare_fn, settings)

    def run_a(self, mod, inputs):
        mod.eval()
        with torch.no_grad():
            return mod(*inputs)

    def run_b(self, mod, inputs):
        mod.eval()
        dll_name = f"{self.name}_{self.curr_iter}.so"
        self.curr_iter += 1
        try:
            mod = self.lower_fn(mod, inputs, self.workdir, self.name, dll_name)
            output = mod(*inputs)
        except RuntimeError as e:
            raise net_min_base.FxNetMinimizerRunFuncError(
                f"Encounter an error when processing \n{mod.graph}\n {e}"
            )
        else:
            return output

    def get_nodes(self, start=None, end=None, enable_print=False):
        nodes = self._collect_nodes(start, end)
        if enable_print:
            _LOGGER.info(f"Nodes fetched from start {start} to end {end} as: {nodes}")
        return nodes
