import torch
from fx2ait.acc_tracer import acc_ops
from fx2ait.tools.common_fx2ait import AITTestCase
from parameterized import parameterized
from torch import nn


class TestSliceTensor(AITTestCase):
    @parameterized.expand(
        [
            ("integer_slice", 1),
            ("slice_batch_dim", slice(None, None, None)),
            ("slice_basic", (slice(None, None, None), slice(0, 3, 1))),
            ("slice_full", (slice(None, None, None), slice(0, 10, 1))),
            ("ellipsis", (slice(None, None, None), ..., slice(0, 3, 1))),
            (
                "slice_all_none",
                (slice(None, None, None), slice(None, None, None)),
            ),
            (
                "slice_start_none",
                (slice(None, None, None), slice(None, 2, 1)),
            ),
            ("slice_end_none", (slice(None, None, None), slice(1, None, 1))),
            (
                "slice_step_none",
                (slice(None, None, None), slice(0, 3, None)),
            ),
            ("slice_neg_idx", (slice(None, None, None), -1)),
            ("slice_neg_slice", (slice(None, None, None), slice(-8, -2, 1))),
            ("multi_dim", (slice(None, None, None), 0, 1)),
            (
                "slice_multi_dim",
                (slice(None, None, None), slice(0, 3, 1), slice(1, -1, 1)),
            ),
            ("none", (slice(None, None, None), None, slice(1, -1, 1), 1)),
            ("with_squeeze", (slice(None, None, None), 1, slice(1, -1, 1), None)),
            (
                "slice_zero_slice",
                (slice(None, None, None), slice(None, None, None), slice(0, 0, None)),
            ),
        ]
    )
    def test_slice_tensor(self, name, idx):
        class SliceTensor(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                y = x + x
                return y[self.idx]

        mod = SliceTensor(idx).half().cuda()
        inputs = [torch.randn(2, 10, 10, 10).half().cuda()]
        self.run_test(mod, inputs, expected_ops={acc_ops.getitem})

    @parameterized.expand([("default", 1), ("neg", -2)])
    def test_get_item(self, _, idx):
        class GetItem(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                shape = x.shape[1:]
                y = torch.nn.functional.layer_norm(x, shape, eps=1e-5)
                return y

        mod = GetItem(idx).half().cuda()
        inputs = [torch.randn(2, 10).half().cuda()]
        self.run_test(
            mod,
            inputs,
            expected_ops={acc_ops.getitem, acc_ops.size, acc_ops.layer_norm},
        )
