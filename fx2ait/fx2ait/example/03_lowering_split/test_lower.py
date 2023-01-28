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
from fx2ait.example.benchmark_utils import benchmark_function
from fx2ait.lower.lower import AitLowerer
from fx2ait.lower.lower_settings import LowerSettings


@torch.fx.wrap
def unsupported_attention_op(f, x):
    attn_out, _ = f(x, x, x)
    return attn_out


class TestFx2aitLowerTests(unittest.TestCase):
    def test_ait_lower(self):
        class LowerModule(torch.nn.Module):
            def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
                super().__init__()
                self.attn = torch.nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                self.linear_net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, dim_feedforward),
                    torch.nn.Dropout(dropout),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(dim_feedforward, dim_feedforward),
                    torch.nn.Linear(dim_feedforward, dim_feedforward),
                    torch.nn.Linear(dim_feedforward, dim_feedforward),
                    torch.nn.Linear(dim_feedforward, dim_feedforward),
                    torch.nn.Linear(dim_feedforward, dim_feedforward),
                    torch.nn.Linear(dim_feedforward, input_dim),
                )
                self.norm1 = torch.nn.LayerNorm(input_dim)
                self.norm2 = torch.nn.LayerNorm(input_dim)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, x):
                # Unsupported op will not be lowered to AIT backend.
                attn_out = unsupported_attention_op(self.attn, x)
                # attn_out, _ = self.attn(x,x,x)
                x = x + self.dropout(attn_out)
                x = self.norm1(x)

                linear_out = self.linear_net(x)
                x = x + self.dropout(linear_out)
                x = self.norm2(x)

                return x

        model = (
            LowerModule(input_dim=768, num_heads=12, dim_feedforward=3072).cuda().half()
        )

        inputs = [torch.randn(10, 196, 768).half().cuda()]

        ref_output = model(*inputs)
        lowerer = AitLowerer.create(
            LowerSettings(
                workdir="/tmp",
                name="test_ait_lower",
                min_acc_module_size=0,
            )
        )
        lowered = lowerer(model, inputs)
        lower_output = lowered(*inputs)

        # Check accuracy
        torch.testing.assert_close(
            ref_output, lower_output, check_dtype=False, atol=1e-2, rtol=1e-2
        )
        # Expect 2 submodules in target model, one is run_on_acc and another run_on_gpu
        children = list(lowered.named_children())
        self.assertEqual(len(children), 2)

        results = []
        for batch_size in [1, 4, 16, 32, 64, 128, 256, 512]:
            inputs = [torch.randn(batch_size, 196, 768).half().cuda()]
            lowered = lowerer(model, inputs)
            results.append(
                benchmark_function(
                    self.__class__.__name__, 100, model, inputs, ait_mod=lowered
                )
            )
        for res in results:
            print(res)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
