import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import param, parameterized


class TestChunkConverter(AITTestCase):
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
        self.run_test(model, inputs, expected_ops={acc_ops.chunk})
