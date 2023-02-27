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
import torchvision
from fx2ait.passes.lower_basic_pass_aten import nchw2nhwc_pass, replace_inplace_ops
from fx2ait.tools.common_aten2ait import DispatchTestCase


class TestModelConverter(DispatchTestCase):
    def test_resnet50(self):
        torch.manual_seed(0)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = torchvision.models.resnet18()

            def forward(self, x):
                return self.mod(x)

        model = TestModule().cuda().half()
        inputs = [torch.randn(32, 3, 224, 224).half().cuda()]
        customized_passes = [
            replace_inplace_ops,
            nchw2nhwc_pass,
        ]

        self.run_test(
            model,
            inputs,
            expected_ops={},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=None,
            customized_passes=customized_passes,
        )

    def test_densenet(self):
        torch.manual_seed(0)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = torchvision.models.densenet121(pretrained=True)

            def forward(self, x):
                return self.mod(x)

        inputs = [torch.randn(1, 3, 224, 224).cuda().half()]
        model = TestModule().cuda().half()
        self.run_test(
            model,
            inputs,
            atol=0.18,
            expected_ops={},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=None,
            customized_passes=[
                replace_inplace_ops,
                nchw2nhwc_pass,
            ],
        )

    # def test_hf_albert_base(self):
    #     # config = AutoConfig.from_pretrained("albert-base-v2")
    #     # config = AutoConfig.from_pretrained("gpt2")
    #     # config = BertConfig()
    #     config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
    #     max_length = 128
    #     batch_size = 32
    #     device = "cuda"

    #     class TestModule(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.mod = AutoModelForMaskedLM.from_config(config).to(device)

    #         def forward(self, x):
    #             return self.mod(x).logits

    #     model = TestModule().cuda().half()
    #     input_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(
    #         device
    #     )
    #     inputs = [input_ids]
    #     self.run_test(
    #         model,
    #         inputs,
    #         expected_ops={},
    #         # permute_inputs=[0, 2, 3, 1],
    #         # permute_outputs=None,
    #         customized_passes=[
    #             replace_inplace_ops,
    #             nchw2nhwc_pass,
    #         ],
    #     )
