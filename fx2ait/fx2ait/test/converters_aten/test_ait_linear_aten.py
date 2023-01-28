import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized


class TestLinearConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("default", [1, 512], True),
            ("matrix", [5, 512], True),
            ("no_bias", [1, 512], False),
            (
                "multi_dim_matrix",
                [4, 5, 512],
                True,
            ),
            (
                "multi_dim_matrix",
                [4, 5, 512],
                False,
            ),
        ]
    )
    def test_linear(self, test_name, shape, bias):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 256, bias)

            def forward(self, x):
                return self.linear(x)

        model = TestModule().cuda().half()
        inputs = [torch.randn(shape).half().cuda()]
        self.run_test(model, inputs, expected_ops={torch.ops.aten.linear})

    @parameterized.expand(
        [
            ("default", [[1, 5], [512, 512]], True),
            ("no_bias", [[1, 4], [512, 512]], False),
            (
                "multi_dim_matrix",
                [[2, 4], [512, 512]],
                True,
            ),
            (
                "multi_dim_matrix_no_bias",
                [[2, 4], [512, 512]],
                False,
            ),
        ]
    )
    def test_dynamic_linear(self, test_name, shape, bias):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 256, bias)

            def forward(self, x):
                return self.linear(x)

        model = TestModule().cuda().half()

        input_shape = []
        for i, s in enumerate(shape):
            input_shape.append(
                TensorSpec.gen_int_var_min_max(s[0], s[1], "dynamic" + str(i))
            )
        input_spec = TensorSpec.create_spec_from_int_vars(
            [input_shape], dtype_list=[torch.float16]
        )

        self.run_test_with_dynamic_shape(
            model, input_spec, expected_ops={torch.ops.aten.linear}
        )
