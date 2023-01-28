import torch
import torchvision
from fx2ait.tools.common_fx2ait import AITTestCase


class TestVisionModelConverter(AITTestCase):
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
        self.run_test(
            model,
            inputs,
            expected_ops={},
            permute_inputs=[0, 2, 3, 1],
            permute_outputs=None,
        )
