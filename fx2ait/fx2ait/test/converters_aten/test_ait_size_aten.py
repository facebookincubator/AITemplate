import torch

# from fx2ait.tensor_spec import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase


class TestATenSizeConverter(DispatchTestCase):
    def test_size(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                t = x.size()
                return y.reshape(t)

        xsize = (2, 3, 4)
        ysize = (2, 12)
        model = TestModule().cuda().half()
        inputs = (torch.randn(xsize).half().cuda(), torch.randn(ysize).half().cuda())

        self.run_test(model, inputs, expected_ops={torch.ops.aten.sym_size})

    ## AIT not support now
    # def test_size_dim(self):
    #     class TestModule(torch.nn.Module):
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             return x.size(1)

    #     size = (2, 3, 4)
    #     model = TestModule().cuda().half()
    #     inputs = (torch.randn(size).half().cuda(),)

    #     self.run_test(model, inputs, expected_ops={torch.ops.aten.sym_size})

    # def test_size_dim_with_dynamic_shape(self):
    #     class TestModule(torch.nn.Module):
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             return x.size(1)

    #     model = TestModule().cuda().half()
    #     inputs_spec = TensorSpec.create_spec_from_shapes(
    #         inputs_min=[
    #             [2, 3, 4],
    #         ],
    #         inputs_max=[
    #             [10, 30, 4],
    #         ],
    #         dtype_list=[
    #             torch.float16,
    #         ],
    #     )

    #     self.run_test_with_dynamic_shape(
    #         model,
    #         inputs_spec,
    #         expected_ops={torch.ops.aten.sym_size},
    #     )
