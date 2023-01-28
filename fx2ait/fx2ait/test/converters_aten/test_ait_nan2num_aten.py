import torch
from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import param, parameterized


class TestCatConverter(DispatchTestCase):
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

        self.run_test(model, inputs, expected_ops={torch.ops.aten.nan_to_num.default})

    @parameterized.expand(
        [
            param("default"),
            param("nan", nan=1.0),
            param("posinf", posinf=1.0),
            param("neginf", neginf=-1.0),
        ]
    )
    def test_dynamic_nan_to_num(self, name, nan=None, posinf=None, neginf=None):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[
                [2, 3, 4],
            ],
            inputs_max=[
                [3, 8, 10],
            ],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.nan_to_num.default},
            specify_num=float("nan"),
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.nan_to_num.default},
            specify_num=float("inf"),
        )

        self.run_test_with_dynamic_shape(
            model,
            inputs_spec,
            expected_ops={torch.ops.aten.nan_to_num.default},
            specify_num=-float("inf"),
        )
