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
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


class SplitGetItemTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SplitGetItemTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_split_getitem(
        self,
        shape,
        split_sections,
        split_dim,
        test_name="split_full_idx",
        dtype="float16",
    ):
        assert len(shape) == 3, f"expected shape to be 3 but got {shape}"
        target = detect_target()
        M, N, K = shape

        X = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[N],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        D = Tensor(
            shape=[M, N],
            dtype=dtype,
            name="input_3",
            is_input=True,
        )
        Y1 = ops.split()(D, split_sections, split_dim)
        Y2 = ops.getitem()(Y1, 0)
        Y = ops.gemm_rcr_bias_sigmoid_mul_tanh()(X, W, B, Y2)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1
        src_ops = set()
        for tensor in module.debug_sorted_graph:
            src_ops |= set(tensor.src_ops())
            for src_op in tensor.src_ops():
                assert not src_op._attrs["op"].startswith("split"), (
                    f"Ecountered split op {src_op}."
                    "Shouldn't have split op after graph optmizaiton"
                )
        assert len(src_ops) == 1

        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        D_pt = get_random_torch_tensor([M, N], dtype)
        Y_pt = torch.tanh(
            torch.sigmoid(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)) * D_pt
        )
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt, "input_3": D_pt}, [y]
        )
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_split_getitem_to_noop(self):
        self._test_split_getitem(
            shape=(16, 32, 10),
            split_sections=[16],
            split_dim=0,
        )
        self._test_split_getitem(
            shape=(16, 32, 10),
            split_sections=[32],
            split_dim=1,
        )

    def _test_split_getitem_remove_output(
        self,
        shape,
        split_sections,
        split_dim,
        test_name="split_remove_output",
        dtype="float16",
    ):
        assert len(shape) == 3, f"expected shape to be 3 but got {shape}"
        target = detect_target()
        M, N, K = shape

        X = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[N],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        D = Tensor(
            shape=[M, N],
            dtype=dtype,
            name="input_3",
            is_input=True,
        )
        Y1 = ops.gemm_rcr_bias_sigmoid_mul_tanh()(X, W, B, D)
        Y2 = ops.split()(Y1, split_sections, split_dim)
        Y = ops.getitem()(Y2, 0)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1
        src_ops = set()
        for tensor in module.debug_sorted_graph:
            src_ops |= set(tensor.src_ops())
            for src_op in tensor.src_ops():
                assert not src_op._attrs["op"].startswith("split"), (
                    f"Ecountered split op {src_op}."
                    "Shouldn't have split op after graph optmizaiton"
                )
        assert len(src_ops) == 1

        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        D_pt = get_random_torch_tensor([M, N], dtype)
        Y_pt = torch.tanh(
            torch.sigmoid(torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)) * D_pt
        )
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(
            {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt, "input_3": D_pt}, [y]
        )
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_split_getitem_remove_output(self):
        self._test_split_getitem_remove_output(
            shape=(16, 32, 10),
            split_sections=[16],
            split_dim=0,
        )
        self._test_split_getitem_remove_output(
            shape=(16, 32, 10),
            split_sections=[32],
            split_dim=1,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
