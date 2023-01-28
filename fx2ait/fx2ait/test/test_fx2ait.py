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
import io
import os
import unittest

import torch
from fx2ait.acc_tracer import acc_tracer
from fx2ait.ait_module import AITModule
from fx2ait.fx2ait import AITInterpreter

torch.ops.load_library("build/libait_model.so")


class TestAITModule(unittest.TestCase):
    def _test_fx2ait_impl(self, test_serialization=False, test_cuda_graph=False):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                sigmoid = torch.sigmoid(x)
                add = sigmoid * sigmoid
                return add

        inputs = [torch.ones(2, 2).cuda().half()]
        mod = TestModule().cuda().half()
        ref_output = mod(*inputs)

        traced = acc_tracer.trace(mod, inputs)

        interp = AITInterpreter(traced, inputs, "./tmp", "test")
        interp_result = interp.run()
        ait_mod = AITModule(
            torch.classes.ait.AITModel(
                interp_result.engine.lib_path,
                interp_result.input_names,
                interp_result.output_names,
                torch.float16,
                torch.float16,
                1,  # num_runtimes
            ),
        )
        ait_mod.engine.use_cuda_graph = test_cuda_graph
        if test_serialization:
            buf = io.BytesIO()
            # Have to JIT-ify the module before we can save/load it.
            ait_mod = torch.jit.trace(ait_mod, inputs)
            script_output = ait_mod(*inputs)
            torch.testing.assert_close(script_output, ref_output, atol=1e-2, rtol=1e-2)
            torch.jit.save(ait_mod, buf)
            buf.seek(0)
            torch.classes.ait.AITModel.register_library_name_to_path_map(
                {
                    os.path.basename(
                        interp_result.engine.lib_path
                    ): interp_result.engine.lib_path
                }
            )
            ait_mod = torch.jit.load(buf)
        ait_output = ait_mod(*inputs)
        torch.testing.assert_close(ait_output, ref_output, atol=1e-2, rtol=1e-2)

    def test_fx2ait(self):
        self._test_fx2ait_impl(test_serialization=False)

    def test_fx2ait_module_serialization(self):
        self._test_fx2ait_impl(test_serialization=True)

    def test_fx2ait_cuda_graph(self):
        self._test_fx2ait_impl(test_cuda_graph=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
