import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestNan2NumConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default"),
            param("nan", nan=1.0),
            param("posinf", posinf=1.0),
            param("neginf", neginf=-1.0),
        ]
    )
    def test_nan_to_num(self, name, nan=None, posinf=None, neginf=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

        model = TestModule().cuda().half()
        inputs = [
            torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14])
            .half()
            .cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.nan_to_num})
