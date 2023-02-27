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
import unittest

import torch
from fx2ait.lower.lower import AitLowerer
from fx2ait.lower.lower_settings import LowerSettings


@torch.fx.wrap
def get_length(input: torch.Tensor) -> int:
    return len(input)


@torch.fx.wrap
def unsupported_op(x):
    return x + x


class TestFx2aitLowerTests(unittest.TestCase):
    def test_fx2ait_lower(self):
        class TestMod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                values = torch.sigmoid(x)
                return get_length(values)

        mod = TestMod().half().cuda()
        input = [torch.randn(2, 3).half().cuda()]
        ref_output = mod(*input)
        lower = AitLowerer.create(
            LowerSettings(workdir="/tmp", name="test_ait_lower", min_acc_module_size=0)
        )
        lowered = lower(mod, input)
        lower_output = lowered(*input)
        self.assertTrue(len(lowered._modules.keys()), 2)
        torch.testing.assert_close(ref_output, lower_output, check_dtype=False)

        # Verify that the resulting module is scriptable and
        # the scripted module is working properly with dynamic batch input
        # TODO: Enable script test after python release include fix:
        # https://github.com/pytorch/pytorch/pull/87804
        # scripted = torch.jit.script(lowered)
        # input2 = [torch.randn(16, 3).half().cuda()]
        # ref_output2 = mod(*input2)
        # torch.testing.assert_close(ref_output2, scripted(*input2), check_dtype=False)

    def test_fx2ait_lower_avoids_copies(self):
        class TestMod(torch.nn.Module):
            def forward(self, x):
                a = unsupported_op(x)
                b = a.unsqueeze(0)
                return unsupported_op(b)

        mod = TestMod().half().cuda()
        x = torch.randn((1,)).half().cuda()
        ref_output = mod(x)
        lowerer = AitLowerer.create(
            LowerSettings(
                workdir="/tmp",
                name="test_ait_lower_avoids_copies",
                min_acc_module_size=0,
            )
        )
        lowered = lowerer(mod, [x])
        lower_output = lowered(x)
        torch.testing.assert_close(ref_output, lower_output, check_dtype=False)

        children = list(lowered.named_children())
        self.assertEqual(len(children), 1)
        name, _ = children[0]
        self.assertNotIn("_run_on_acc", name)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
