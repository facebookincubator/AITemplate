import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestExpandConverter(AITTestCase):
    @parameterized.expand(
        [
            param("same_shapes", [1, 2, 3], [1, 2, 3]),
            param("infer_shapes", [1, 2, 3], [-1, -1, -1]),
        ]
    )
    def test_expand(self, name, orig_shape, target_shape):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                y = x.expand(target_shape)
                return y * y

        class TestModuleManyArgs(torch.nn.Module):
            def forward(self, x):
                y = x.expand(*target_shape)
                return y * y

        model = TestModule().cuda().half()
        inputs = [torch.randn(orig_shape).half().cuda()]
        self.run_test(model, inputs, expected_ops={acc_ops.expand})

        model_many_args = TestModuleManyArgs().cuda().half()
        self.run_test(model_many_args, inputs, expected_ops={acc_ops.expand})
