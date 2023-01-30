import torch
import torch.nn as nn
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


class TestFlattenConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("flatten_middle_dims", 1, 2),
            ("flatten_last_3_dims", 1, 3),
            ("flatten_all", 0, 3),
        ]
    )
    def test_flatten(self, _, start_dim, end_dim):
        class TestModule(nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                return torch.flatten(x, self.start, self.end)

        model = TestModule(start_dim, end_dim).cuda().half()
        inputs = (torch.randn(1, 2, 3, 1).half().cuda(),)

        self.run_test(model, inputs, expected_ops={torch.ops.aten.view.default})

    @parameterized.expand(
        [
            ("flatten_middle_dims", 1, 2),
            ("flatten_last_3_dims", 1, 3),
        ]
    )
    def test_flatten_with_dynamic_shape(self, _, start_dim, end_dim):
        class TestModule(nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                return torch.flatten(x, self.start, self.end)

        model = TestModule(start_dim, end_dim).cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [1, 2, 3, 4],
            ],
            inputs_max=[
                [10, 20, 3, 4],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.view.default},
        )
