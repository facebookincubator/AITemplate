#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
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
            (
                "unsqueeze_inner_dim_twice",
                (
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, None, None),
                    None,
                    None,
                ),
            ),
            ("with_squeeze", (slice(None, None, None), 1, slice(1, -1, 1), None)),
            (
                "slice_zero_slice",
                (slice(None, None, None), slice(None, None, None), slice(0, 0, None)),
            ),
            (
                "slice_start_seq_slice",
                (slice(0, 1, None), [0, 1, 2], slice(0, 10, None)),
            ),
            (
                "slice_end_seq_slice",
                (slice(0, 1, None), [0, 6, 7, 8, 9], slice(0, 10, None)),
            ),
            (
                "slice_long_seq_slice",
                (slice(0, 1, None), [0, 5, 6, 7, 2, 3, 4, 5], slice(0, 10, None)),
            ),
            (
                "slice_list_slice",
                (slice(0, 1, None), [2], slice(0, 10, None)),
            ),
            (
                "zero_list_zero",
                (slice(0, 1, None), [0, 7, 5, 3, 1, 9], slice(0, 0, None)),
            ),
            (
                "all_list_all",
                (slice(None, None, None), [2, 2, 2, 2], slice(None, None, None)),
            ),
            (
                "slice_zero_list",
                (slice(0, 1, None), slice(0, 0, None), [0, 1, 3]),
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
