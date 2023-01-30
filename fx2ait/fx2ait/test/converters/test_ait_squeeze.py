import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestSqueezeConverter(AITTestCase):
    @parameterized.expand(
        [
            param("default", dim=None, shape=[2, 1, 1, 3]),
            param("1", dim=1, shape=[2, 1, 1, 3]),
            param("-1", dim=-1, shape=[2, 1, 3, 1]),
        ]
    )
    def test_squeeze(self, name, dim, shape):
        class TestModule(torch.nn.Module):
            def forward(self, y: torch.Tensor) -> torch.Tensor:
                squeeze = (
                    torch.squeeze(y, dim=dim) if dim is not None else torch.squeeze(y)
                )
                return squeeze

        model = TestModule().cuda().half()
        inputs = [
            torch.randn(shape).half().cuda(),
        ]

        self.run_test(model, inputs, expected_ops={acc_ops.squeeze})
