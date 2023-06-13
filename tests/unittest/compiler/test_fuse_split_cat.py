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

import torch
from aitemplate.compiler import compile_model, ops

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.public import IntImm

from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor, graph_has_op


class FuseSplitCatTestCase(unittest.TestCase):
    def _test_fuse_split_cat_rearrange(self, M, N, split, remove_split=True):
        dtype = "float16"
        M = IntImm(M)
        N = IntImm(N)

        input_1 = Tensor(
            shape=[M, N],
            name="input_1",
            is_input=True,
        )
        split_2 = ops.split()(input_1, split, 0)
        concatenate_3 = ops.concatenate()(split_2[::-1], 0)

        # Set outputs
        concatenate_3._attrs["name"] = "output_0"
        concatenate_3._attrs["is_output"] = True
        # Compile
        model = compile_model(
            concatenate_3, detect_target(), "./tmp", self._testMethodName
        )
        # Check that split was removed
        self.assertEqual(
            graph_has_op(model.debug_sorted_graph, "split"), not remove_split
        )
        # Run
        input_1 = get_random_torch_tensor((M.value(), N.value()), dtype=dtype)
        # Compare
        split_pt = torch.split(input_1, split, 0)
        y_pt = torch.cat(
            [split_pt[1], split_pt[0]],
            0,
        )
        y_ait = torch.empty_like(y_pt)
        model.run_with_tensors(
            {"input_1": input_1},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)

    def test_fuse_split_cat_even(self):
        self._test_fuse_split_cat_rearrange(
            512, 512, split=[256, 256], remove_split=True
        )

    def test_fuse_split_cat_odd(self):
        self._test_fuse_split_cat_rearrange(
            512, 512, split=[139, 373], remove_split=True
        )

    def test_fuse_split_cat_reuse(self):
        """Use a split output twice in the concatenate op."""
        dtype = "float16"
        M = IntImm(512)
        N = IntImm(512)

        input_1 = Tensor(
            shape=[M, N],
            name="input_1",
            is_input=True,
        )
        split_2 = ops.split()(input_1, int(M.value() / 2), 0)
        concatenate_3 = ops.concatenate()([split_2[1], split_2[0], split_2[1]], 0)

        # Set outputs
        concatenate_3._attrs["name"] = "output_0"
        concatenate_3._attrs["is_output"] = True
        # Compile
        model = compile_model(
            concatenate_3, detect_target(), "./tmp", self._testMethodName
        )
        # Check that split was removed
        self.assertFalse(graph_has_op(model.debug_sorted_graph, "split"))
        # Run
        input_1 = get_random_torch_tensor((M.value(), N.value()), dtype=dtype)
        # Compare
        split_pt = torch.split(input_1, int(M.value() / 2), 0)
        y_pt = torch.cat(
            [split_pt[1], split_pt[0], split_pt[1]],
            0,
        )
        y_ait = torch.empty_like(y_pt)
        model.run_with_tensors(
            {"input_1": input_1},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)

    def test_fuse_split_cat_dim1(self):
        dtype = "float16"
        M = IntImm(512)
        N = IntImm(512)

        input_1 = Tensor(
            shape=[M, N],
            name="input_1",
            is_input=True,
        )
        split_2 = ops.split()(input_1, int(N.value() / 2), 1)
        concatenate_3 = ops.concatenate()(split_2[::-1], 1)

        # Set outputs
        concatenate_3._attrs["name"] = "output_0"
        concatenate_3._attrs["is_output"] = True
        # Compile
        model = compile_model(
            concatenate_3, detect_target(), "./tmp", self._testMethodName
        )
        # Check that split was removed
        self.assertFalse(graph_has_op(model.debug_sorted_graph, "split"))
        # Run
        input_1 = get_random_torch_tensor((M.value(), N.value()), dtype=dtype)
        # Compare
        split_pt = torch.split(input_1, int(N.value() / 2), 1)
        y_pt = torch.cat(
            split_pt[::-1],
            1,
        )
        y_ait = torch.empty_like(y_pt)
        model.run_with_tensors(
            {"input_1": input_1},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)

    def test_fuse_split_cat_different_dims(self):
        """Splitting and then concatting on different dims is not
        expected to be optimized currently."""
        dtype = "float16"
        M = IntImm(512)
        N = IntImm(512)

        input_1 = Tensor(
            shape=[M, N],
            name="input_1",
            is_input=True,
        )
        split_2 = ops.split()(input_1, int(M.value() / 2), 0)
        concatenate_3 = ops.concatenate()(split_2[::-1], 1)

        # Set outputs
        concatenate_3._attrs["name"] = "output_0"
        concatenate_3._attrs["is_output"] = True
        # Compile
        model = compile_model(
            concatenate_3, detect_target(), "./tmp", self._testMethodName
        )
        # Check that split was not removed because the dims are different
        self.assertTrue(graph_has_op(model.debug_sorted_graph, "split"))
        # Run
        input_1 = get_random_torch_tensor((M.value(), N.value()), dtype=dtype)
        # Compare
        split_pt = torch.split(input_1, int(M.value() / 2), 0)
        y_pt = torch.cat(
            split_pt[::-1],
            1,
        )
        y_ait = torch.empty_like(y_pt)
        model.run_with_tensors(
            {"input_1": input_1},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)

    def test_fuse_split_cat_bmm(self):
        """Optimize out a split op whose output is used by both concat and bmm."""
        dtype = "float16"
        B = 1
        M = 128
        N = 512
        K = 512
        split_size_or_sections = 256
        split_dim = 2
        T_A = Tensor(
            # feed the second half of T_A into additional concat so that the split
            # output is used by both bmm and concat
            shape=[B, M, K * 2],
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        T_B = Tensor(
            shape=[B, N, K],
            dtype=dtype,
            name="input1",
            is_input=True,
        )

        Xs = ops.split()(T_A, split_size_or_sections, split_dim)
        Ys = ops.split()(T_B, split_size_or_sections, split_dim)
        assert len(Xs) // 2 == len(Ys)

        n = 2
        Cs = []
        for i in range(n):
            X = Xs[i]
            Y = Ys[i]
            C = ops.bmm_rcr()(X, Y)
            Cs.append(C)
        # do an extra concatenate so that split_1 has different output ops
        extra_concat = ops.concatenate()([Xs[3], Xs[2], Xs[3], Xs[2]], dim=split_dim)
        bmm_cat = ops.concatenate()(Cs, dim=split_dim)
        Y = ops.elementwise(FuncEnum.ADD)(extra_concat, bmm_cat)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        a = get_random_torch_tensor([B, M, K * 2], dtype)
        b = get_random_torch_tensor([B, N, K], dtype)
        xs = a.split(split_size_or_sections, split_dim)
        ys = b.split(split_size_or_sections, split_dim)
        cs = []
        for i in range(n):
            x = xs[i]
            y = ys[i]
            c = torch.bmm(x, y.permute(0, 2, 1))
            cs.append(c)
        extra_concat_pt = torch.cat([xs[3], xs[2], xs[3], xs[2]], dim=split_dim)
        bmm_cat_pt = torch.cat(cs, dim=split_dim)
        y_pt = torch.add(extra_concat_pt, bmm_cat_pt)

        # Gen module.
        target = detect_target()
        model = compile_model(Y, target, "./tmp", self._testMethodName)
        # Both splits should be removed, including the split that is used by
        # both bmm and concat
        self.assertFalse(graph_has_op(model.debug_sorted_graph, "split"))
        self.assertEqual(len(model.debug_sorted_graph), 5)
        y = torch.empty_like(y_pt)
        model.run_with_tensors({"input0": a, "input1": b}, [y])
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
