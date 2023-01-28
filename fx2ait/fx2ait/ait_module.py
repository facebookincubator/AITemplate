from typing import List

import torch


class AITModule(torch.nn.Module):
    def __init__(
        self,
        engine=None,
    ):
        super(AITModule, self).__init__()
        self.engine = engine

    def forward(self, *inputs):
        outputs = self.engine.forward(inputs)
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
    def create_ait_module_wrapper(engine, trace_ait_module, *inputs):
        """
        Some use cases need to torch.jit.script a model with AITModules in
        it, but TorchScript does not support variadic inputs. We can get
        around this by scripting the AITModule with some sample inputs.
        This is turned in by passing allow_scripting=True.
        """
        mod = AITModule(engine)
        return torch.jit.trace(mod, inputs) if trace_ait_module else mod
