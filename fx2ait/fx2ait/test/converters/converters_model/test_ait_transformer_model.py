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
from fx2ait.tools.common_fx2ait import AITTestCase


class TestTransformerModelConverter(AITTestCase):
    def test_transformer_encoder(self):
        torch.manual_seed(0)

        class EncoderBlock(torch.nn.Module):
            def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
                """
                Inputs:
                    input_dim - Dimensionality of the input
                    num_heads - Number of heads to use in the attention block
                    dim_feedforward - Dimensionality of the hidden layer in the MLP
                    dropout - Dropout probability to use in the dropout layers
                """
                super().__init__()
                # Attention layer
                self.attn = torch.nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                # # Two-layer MLP
                self.linear_net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, dim_feedforward),
                    torch.nn.Dropout(dropout),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(dim_feedforward, input_dim),
                )
                # Layers to apply in between the main layers
                self.norm1 = torch.nn.LayerNorm(input_dim)
                self.norm2 = torch.nn.LayerNorm(input_dim)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, x):
                # Attention part
                attn_out, _ = self.attn(query=x, key=x, value=x)
                # return attn_out
                x = x + self.dropout(attn_out)
                x = self.norm1(x)

                # MLP part
                linear_out = self.linear_net(x)
                x = x + self.dropout(linear_out)
                x = self.norm2(x)

                return x

        model = (
            EncoderBlock(input_dim=512, num_heads=16, dim_feedforward=12).cuda().half()
        )

        inputs = [torch.randn(10, 32, 512).half().cuda()]
        self.run_test(
            model,
            inputs,
            expected_ops={},
        )
