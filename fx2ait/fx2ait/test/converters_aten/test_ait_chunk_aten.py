import torch
from fx2ait.fx2ait import TensorSpec
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import param, parameterized


class TestChunkConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("default", 2, [3, 10, 2], 1),
            param("no_dim", 2, [3, 10, 2]),
            param("neg_dim", 1, [3, 10, 2], -2),
            param("chunk_bigger_than_dim", 4, [2, 10, 2], 2),
        ]
    )
    def test_chunk(self, name, chunks, shape, dim=None):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                x = (
                    torch.chunk(x, chunks=chunks, dim=dim)
                    if dim is not None
                    else torch.chunk(x, chunks=chunks)
                )
                # For AIT, all chunk results must be used
                return x[0]

        model = TestModule().cuda().half()
        inputs = [torch.randn(shape).half().cuda()]
        self.run_test(model, inputs, expected_ops={})

    def test_chunk_dynamic(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                x = torch.chunk(x, chunks=2, dim=1)
                return x[0]

        model = TestModule().cuda().half()
        inputs_spec = TensorSpec.create_spec_from_shapes(
            inputs_min=[[20, 10, 8]],
            inputs_max=[[50, 10, 8]],
            dtype_list=[
                torch.float16,
            ],
        )

        self.run_test_with_dynamic_shape(model, inputs_spec, expected_ops={})
