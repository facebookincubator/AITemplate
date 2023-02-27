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
import tempfile
import unittest

import torch
from fx2ait.acc_tracer import acc_tracer
from fx2ait.ait_module import AITModule
from fx2ait.fx2ait import AITInterpreter

OSS_AIT_MODEL = False
try:
    torch.ops.load_library("//deeplearning/ait:AITModel")
except Exception:
    torch.ops.load_library("build/libait_model.so")
    OSS_AIT_MODEL = True

AIT_MODEL_CLASS = (
    torch.classes.ait.AITModel if OSS_AIT_MODEL else torch.classes.fb.AITModel
)


class TestAITModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_fx2ait_impl(self, test_serialization=False, test_cuda_graph=False):
        mod = (
            torch.nn.Sequential(
                torch.nn.Linear(3, 4),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
                torch.nn.ReLU(),
            )
            .half()
            .cuda()
        )
        inputs = [torch.randn(5, 3).half().cuda()]
        ref_output = mod(*inputs)

        traced = acc_tracer.trace(mod, inputs)

        ait_dump_dir = tempfile.mkdtemp(prefix="test_fx2ait_", dir="/tmp")

        interp = AITInterpreter(traced, inputs, ait_dump_dir, "test")
        interp_result = interp.run()
        ait_mod = AITModule(
            AIT_MODEL_CLASS(
                interp_result.engine.lib_path,
                interp_result.input_names,
                interp_result.output_names,
                torch.float16,
                torch.float16,
                1,  # num_runtimes
            )
        )
        ait_mod.engine.use_cuda_graph = test_cuda_graph
        if test_serialization:
            buf = io.BytesIO()
            # Have to JIT-ify the module before we can save/load it.
            ait_mod = torch.jit.trace(ait_mod, inputs)
            script_output = ait_mod(*inputs)
            torch.testing.assert_close(script_output, ref_output, atol=0.1, rtol=0.1)
            torch.jit.save(ait_mod, buf)
            buf.seek(0)
            AIT_MODEL_CLASS.register_library_name_to_path_map(
                {
                    os.path.basename(
                        interp_result.engine.lib_path
                    ): interp_result.engine.lib_path
                }
            )
            ait_mod = torch.jit.load(buf)
        ait_output = ait_mod(*inputs)
        torch.testing.assert_close(ait_output, ref_output, atol=0.1, rtol=0.1)
        if not OSS_AIT_MODEL:
            weights = {
                "_0_weight": torch.ones(3, 4).cuda().half(),
                "_0_bias": torch.randn(4).cuda().half(),
            }
            ait_mod.engine.update_constants_with_weights(weights)
            ait_output = ait_mod(*inputs)
            torch.testing.assert_close(ait_output, ref_output, atol=1e-2, rtol=1e-2)
            ait_mod.engine.swap_constants()
            ait_output = ait_mod(*inputs)
            self.assertFalse(
                torch.allclose(ait_output, ref_output, atol=1e-2, rtol=1e-2)
            )

    def test_fx2ait(self):
        self._test_fx2ait_impl(test_serialization=False)

    def test_fx2ait_module_serialization(self):
        self._test_fx2ait_impl(test_serialization=True)

    def test_fx2ait_cuda_graph(self):
        self._test_fx2ait_impl(test_cuda_graph=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
