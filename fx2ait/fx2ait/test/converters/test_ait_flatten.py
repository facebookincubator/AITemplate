import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestFlattenConverter(AITTestCase):
    @parameterized.expand(
        [param("default"), param("start", start_dim=1), param("end", end_dim=2)]
    )
    def test_clamp(self, name, start_dim=0, end_dim=-1):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(1, 2, 3).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.flatten})
