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
from fx2ait.passes.lower_basic_pass_aten import (
    aten_compose_getitem_slice,
    compose_getitem_slice,
)
from fx2ait.tools.common_aten2ait import DispatchTestCase
from parameterized import parameterized
from torch import nn


class TestSliceTensor(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "integer_slice",
                1,
                {
                    torch.ops.aten.select.int,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_batch_dim",
                slice(None, None, None),
                {
                    torch.ops.aten.slice.Tensor,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_basic",
                (slice(None, None, None), slice(0, 3, 1)),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_full",
                (slice(None, None, None), slice(0, 10, 1)),
                {
                    torch.ops.aten.slice.Tensor,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            ## Trace problem in support of ellipsis
            # (
            #     "ellipsis",  # It seems there is some problem in tracing ellipsis: P539875442
            #     (slice(None, None, None), ..., slice(0, 3, 1)),
            #     {
            #         torch.ops.aten.add.Tensor,
            #     },
            # ),
            (
                "slice_all_none",
                (
                    slice(None, None, None),
                    slice(None, None, None),
                ),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_start_none",
                (
                    slice(None, None, None),
                    slice(None, 2, 1),
                ),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_end_none",
                (slice(None, None, None), slice(1, None, 1)),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_step_none",
                (
                    slice(None, None, None),
                    slice(0, 3, None),
                ),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_neg_idx",
                (slice(None, None, None), -1),
                {
                    torch.ops.aten.slice.Tensor,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_neg_slice",
                (slice(None, None, None), slice(-8, -2, 1)),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "multi_dim",
                (slice(None, None, None), 0, 1),
                {
                    torch.ops.aten.slice.Tensor,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_multi_dim",
                (slice(None, None, None), slice(0, 3, 1), slice(1, -1, 1)),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "none",
                (slice(None, None, None), None, slice(1, -1, 1), 1),
                {
                    torch.ops.aten.slice.Tensor,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "with_squeeze",
                (slice(None, None, None), 1, slice(1, -1, 1), None),
                {
                    torch.ops.aten.slice.Tensor,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_zero_slice",
                (slice(None, None, None), slice(None, None, None), slice(0, 0, None)),
                {
                    aten_compose_getitem_slice,
                    torch.ops.aten.add.Tensor,
                },
                None,
            ),
            (
                "slice_basic_compose",
                (slice(None, None, None), slice(None, None, None), slice(0, 3, 1)),
                {
                    torch.ops.aten.add.Tensor,
                    aten_compose_getitem_slice,
                },
                [
                    compose_getitem_slice,
                ],
            ),
            (
                "slice_zero_slice_compose",
                (slice(None, None, None), slice(None, None, None), slice(0, 0, None)),
                {
                    torch.ops.aten.add.Tensor,
                    aten_compose_getitem_slice,
                },
                [
                    compose_getitem_slice,
                ],
            ),
        ]
    )
    def test_slice_tensor(self, name, idx, expected_ops, customized_passes):
        class SliceTensor(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                y = x + x
                return y[self.idx]

        mod = SliceTensor(idx).half().cuda()

        inputs = [torch.randn(2, 10, 10, 10).half().cuda()]
        self.run_test(
            mod,
            inputs,
            expected_ops=expected_ops,
            customized_passes=customized_passes,
        )
