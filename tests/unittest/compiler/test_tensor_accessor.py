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

import unittest
from typing import List, Optional

from aitemplate.compiler.base import IntVar, Tensor
from aitemplate.compiler.tensor_accessor import TensorAccessor


class TensorAccessorTestCase(unittest.TestCase):
    def test_dim_mapping_for_stride(self):
        tc = TensorAccessor(Tensor(shape=[2, 2, 4]))
        self.assertListEqual(tc._dim_mapping, [([0], [0]), ([1], [1]), ([2], [2])])

        tc.update_base_tensor(
            Tensor(shape=[2, 2, 4]), stride_dim=2, stride_dim_offset=10
        )
        self.assertListEqual(tc._dim_mapping, [([0], [0]), ([1], [1]), ([2], [2])])

    def _test_dim_mapping_helper(
        self, original_shape, new_shape, expected_dim_mapping, test_reverse=False
    ):
        tc = TensorAccessor(Tensor(shape=original_shape))
        tc.update_base_tensor_shape(Tensor(shape=new_shape))
        self.assertEqual(tc._dim_mapping, expected_dim_mapping)

        if test_reverse:
            tc = TensorAccessor(Tensor(shape=new_shape))
            tc.update_base_tensor_shape(Tensor(shape=original_shape))
            expected_dim_mapping = (
                [(mapping[1], mapping[0]) for mapping in expected_dim_mapping]
                if expected_dim_mapping is not None
                else None
            )
            self.assertEqual(tc._dim_mapping, expected_dim_mapping)

    def test_dim_mapping_for_reshape_basic(self):
        self._test_dim_mapping_helper(
            original_shape=[2, 2, 4],
            new_shape=[2, 2, 4],
            expected_dim_mapping=[([0], [0]), ([1], [1]), ([2], [2])],
            test_reverse=True,
        )
        self._test_dim_mapping_helper(
            original_shape=[2, 2, 4],
            new_shape=[2, 8],
            expected_dim_mapping=[([0], [0]), ([1, 2], [1])],
            test_reverse=True,
        )
        self._test_dim_mapping_helper(
            original_shape=[2, 2, 4],
            new_shape=[4, 4],
            expected_dim_mapping=[([0, 1], [0]), ([2], [1])],
            test_reverse=True,
        )
        self._test_dim_mapping_helper(
            original_shape=[2, 2, 4],
            new_shape=[16],
            expected_dim_mapping=[([0, 1, 2], [0])],
            test_reverse=True,
        )
        self._test_dim_mapping_helper(
            original_shape=[2, 2, 4],
            new_shape=[8, 2],
            expected_dim_mapping=[([0, 1, 2], [0, 1])],
            test_reverse=True,
        )

    def test_dim_mapping_for_reshape_ones(self):
        self._test_dim_mapping_helper(
            original_shape=[1, 1, 1, 1],
            new_shape=[1, 1, 1],
            expected_dim_mapping=[
                ([0], []),
                ([1], []),
                ([2], []),
                ([3], []),
                ([], [0]),
                ([], [1]),
                ([], [2]),
            ],
            test_reverse=False,
        )
        self._test_dim_mapping_helper(
            original_shape=[1, 3, 1, 1, 5, 1],
            new_shape=[1, 3, 5, 1, 1],
            expected_dim_mapping=[
                ([0], []),
                ([], [0]),
                ([1], [1]),
                ([2], []),
                ([3], []),
                ([4], [2]),
                ([5], []),
                ([], [3]),
                ([], [4]),
            ],
            test_reverse=False,
        )

    def test_dim_mapping_for_reshape_dynamic(self):
        batch_dim = IntVar([1, 2], name="batch_size")
        self._test_dim_mapping_helper(
            original_shape=[batch_dim, 2, 4],
            new_shape=[batch_dim, 2, 4],
            expected_dim_mapping=[([0], [0]), ([1], [1]), ([2], [2])],
            test_reverse=True,
        )
        self._test_dim_mapping_helper(
            original_shape=[batch_dim, 2, 4],
            new_shape=[batch_dim, 8],
            expected_dim_mapping=[([0], [0]), ([1, 2], [1])],
            test_reverse=True,
        )
        self._test_dim_mapping_helper(
            original_shape=[batch_dim, 2, 4],
            new_shape=[4, batch_dim],
            expected_dim_mapping=None,
            test_reverse=True,
        )
        self._test_dim_mapping_helper(
            original_shape=[batch_dim, 2, 4],
            new_shape=[1, 1, batch_dim, 8],
            expected_dim_mapping=[([], [0]), ([], [1]), ([0], [2]), ([1, 2], [3])],
            test_reverse=True,
        )

        self._test_dim_mapping_helper(
            original_shape=[1, 1, 1, batch_dim, 1, 1],
            new_shape=[batch_dim],
            expected_dim_mapping=[
                ([0], []),
                ([1], []),
                ([2], []),
                ([3], [0]),
                ([4], []),
                ([5], []),
            ],
            test_reverse=True,
        )

    def _test_get_stride_str_helper(
        self,
        original_shape,
        view_shape: Optional[str],
        stride_shape: str,
        stride_dim,
        dim,
        dim_names: Optional[List[str]],
        expected_stride_strs,
    ):
        tc = TensorAccessor(Tensor(shape=original_shape))
        if view_shape is not None:
            tc.update_base_tensor_shape(Tensor(shape=view_shape))
        tc.update_base_tensor(
            Tensor(shape=stride_shape), stride_dim, stride_dim_offset=0
        )
        self.assertEqual(tc.try_get_stride_strs(dim, dim_names), expected_stride_strs)

    def test_stride_strs_basic(self):
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=None,
            stride_shape=[2, 8, 2],
            stride_dim=1,
            dim=2,
            dim_names=None,
            expected_stride_strs=[],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=None,
            stride_shape=[2, 8, 2],
            stride_dim=1,
            dim=1,
            dim_names=None,
            expected_stride_strs=["2"],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=None,
            stride_shape=[2, 8, 2],
            stride_dim=1,
            dim=0,
            dim_names=None,
            expected_stride_strs=["8", "2"],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=[2, 4, 2],
            stride_shape=[2, 8, 2],
            stride_dim=1,
            dim=0,
            dim_names=None,
            expected_stride_strs=["8", "2"],
        )

    def test_stride_strs_static_mapping(self):
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=[2, 8],
            stride_shape=[2, 16],
            stride_dim=1,
            dim=0,
            dim_names=None,
            expected_stride_strs=["16"],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=[2, 8],
            stride_shape=[2, 16],
            stride_dim=1,
            dim=1,
            dim_names=None,
            expected_stride_strs=None,
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=[2, 8],
            stride_shape=[2, 16],
            stride_dim=1,
            dim=2,
            dim_names=None,
            expected_stride_strs=[],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 2, 4],
            view_shape=[2, 2, 2, 2],
            stride_shape=[2, 2, 2, 4],
            stride_dim=3,
            dim=1,
            dim_names=None,
            expected_stride_strs=None,
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 2, 4],
            view_shape=[2, 2, 2, 2],
            stride_shape=[2, 2, 4, 2],
            stride_dim=2,
            dim=1,
            dim_names=None,
            expected_stride_strs=["4", "2"],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=[2, 8, 1, 1],
            stride_shape=[2, 8, 1, 2],
            stride_dim=3,
            dim=2,
            dim_names=None,
            expected_stride_strs=["2"],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=[2, 8, 1, 1],
            stride_shape=[2, 8, 1, 2],
            stride_dim=3,
            dim=1,
            dim_names=["a", "b", "c"],
            expected_stride_strs=["2", "2"],
        )
        self._test_get_stride_str_helper(
            original_shape=[2, 4, 2],
            view_shape=[2, 8, 1, 1],
            stride_shape=[2, 8, 1, 2],
            stride_dim=3,
            dim=0,
            dim_names=None,
            expected_stride_strs=["4", "2", "2"],
        )

    def test_stride_strs_dynamic_mapping(self):
        batch_dim = IntVar([1, 2], name="batch_size")
        emb_dim = IntVar([1, 2], name="emb_dim")
        self._test_get_stride_str_helper(
            original_shape=[batch_dim, emb_dim, 2],
            view_shape=None,
            stride_shape=[batch_dim, emb_dim, 4],
            stride_dim=2,
            dim=0,
            dim_names=["a", "b", "c"],
            expected_stride_strs=["b", "4"],
        )
        self._test_get_stride_str_helper(
            original_shape=[batch_dim, emb_dim, 2],
            view_shape=[batch_dim, emb_dim, 2],
            stride_shape=[batch_dim, emb_dim, 4],
            stride_dim=2,
            dim=0,
            dim_names=["a", "b", "c"],
            expected_stride_strs=["b", "4"],
        )

        self._test_get_stride_str_helper(
            original_shape=[batch_dim, 4, 2],
            view_shape=[batch_dim, 8],
            stride_shape=[batch_dim, 16],
            stride_dim=1,
            dim=0,
            dim_names=["a", "b", "c"],
            expected_stride_strs=["16"],
        )
        self._test_get_stride_str_helper(
            original_shape=[batch_dim, 4, 2],
            view_shape=[batch_dim, 8],
            stride_shape=[batch_dim, 16],
            stride_dim=1,
            dim=1,
            dim_names=None,
            expected_stride_strs=None,
        )
        self._test_get_stride_str_helper(
            original_shape=[batch_dim, 4, 2],
            view_shape=[batch_dim, 8],
            stride_shape=[batch_dim, 16],
            stride_dim=1,
            dim=2,
            dim_names=None,
            expected_stride_strs=[],
        )
        self._test_get_stride_str_helper(
            original_shape=[batch_dim, 4, 2],
            view_shape=[batch_dim, 8, 1, 1],
            stride_shape=[batch_dim, 8, 1, 2],
            stride_dim=3,
            dim=2,
            dim_names=None,
            expected_stride_strs=["2"],
        )
        self._test_get_stride_str_helper(
            original_shape=[batch_dim, 8],
            view_shape=[batch_dim, 4, 2],
            stride_shape=[batch_dim, 4, 4],
            stride_dim=2,
            dim=0,
            dim_names=["a", "b"],
            expected_stride_strs=None,
        )
        self._test_get_stride_str_helper(
            original_shape=[batch_dim, 8],
            view_shape=[batch_dim, 4, 2],
            stride_shape=[batch_dim, 4, 4],
            stride_dim=2,
            dim=1,
            dim_names=["a", "b"],
            expected_stride_strs=[],
        )
