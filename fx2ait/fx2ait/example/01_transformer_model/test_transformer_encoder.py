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
from fx2ait.example.benchmark_utils import benchmark_function, verify_accuracy


class TestTransformerModule(unittest.TestCase):
    def test_transformer_encoder(self):
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
            EncoderBlock(input_dim=768, num_heads=12, dim_feedforward=3072)
            .cuda()
            .half()
        )

        inputs = [torch.randn(10, 196, 768).half().cuda()]
        verify_accuracy(model, inputs)

        results = []
        for batch_size in [1, 4, 16, 32, 64, 128, 256, 512]:
            inputs = [torch.randn(batch_size, 196, 768).half().cuda()]
            results.append(
                benchmark_function(self.__class__.__name__, 100, model, inputs)
            )
        for res in results:
            print(res)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
