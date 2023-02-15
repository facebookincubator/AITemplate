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
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase


class TestMultiHeadAttentionConverter(AITTestCase):
    def test_multihead_attention_cross_attenytion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, dim, nheads):
                super().__init__()
                self.attn = torch.nn.modules.activation.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=nheads,
                    batch_first=True,
                )

            def forward(self, x):
                layer_norm = torch.nn.functional.layer_norm(x, (dim,), eps=1e-5)
                getitem = layer_norm[slice(None, None, None), 0]
                unsqueeze = torch.unsqueeze(getitem, dim=1)

                return self.attn(query=unsqueeze, key=layer_norm, value=layer_norm)

        seq_len_q, dim, nheads = 4, 256, 16
        model = TestModule(dim, nheads).half().cuda()
        input_q = torch.randn(128, seq_len_q, dim).cuda().half()
        self.run_test(
            model,
            [input_q],
            expected_ops={
                torch.nn.modules.activation.MultiheadAttention,
                acc_ops.layer_norm,
                acc_ops.unsqueeze,
                acc_ops.getitem,
            },
            leaf_module=torch.nn.MultiheadAttention,
        )

    def test_multihead_attention(self):
        class TestModule(torch.nn.Module):
            def __init__(self, dim, nheads):
                super().__init__()
                self.attn = torch.nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=nheads,
                    batch_first=True,
                )

            def forward(self, x):
                return self.attn(query=x, key=x, value=x)

        batch_size = 2
        seqlen = 4
        dim = 512
        num_heads = 8

        x = torch.ones(batch_size, seqlen, dim).cuda().half()
        model = TestModule(dim, num_heads).eval().half().cuda()

        self.run_test(
            model,
            [x],
            expected_ops={torch.nn.MultiheadAttention},
            leaf_module=torch.nn.MultiheadAttention,
        )
