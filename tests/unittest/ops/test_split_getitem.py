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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from aitemplate.utils import shape_utils
from parameterized import parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class SplitGetItemTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SplitGetItemTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_split_getitem(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32),
        split_sections=(4, 8, 2, 2),
        split_dim=1,
        item_idx=0,
        test_name="split_getitem",
        dtype="float16",
    ):
        assert len(X_shape) == 2, f"expected X_shape to be 2 but got {X_shape}"
        target = detect_target()
        b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
        X = Tensor(
            shape=[b_dim, *X_shape],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        N = 16
        if split_dim == 1:
            K = X_shape[1]
        elif split_dim == 2:
            K = split_sections[item_idx]
        else:
            assert 0, f"expected split_dim to be either 1 or 2 but got {split_dim}"

        W = Tensor(
            shape=[b_dim, N, K],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )

        Y1 = ops.split()(X, split_sections, split_dim)
        Y2 = ops.getitem()(Y1, item_idx)
        Y = ops.bmm_rcr()(Y2, W)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = get_random_torch_tensor(X_shape_pt, dtype=dtype)
            W_pt = get_random_torch_tensor([b, N, K], dtype=dtype)
            WT = torch.transpose(W_pt, 2, 1)

            Y1_pt = torch.split(X_pt, split_sections, split_dim)
            Y_pt = torch.bmm(Y1_pt[item_idx], WT)

            y = torch.empty_like(Y_pt)
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32")],
            }
        )
    )
    def test_split_getitem(self, dtype):
        self._test_split_getitem(
            test_name=f"split_getitem_{dtype}",
            dtype=dtype,
        )
        self._test_split_getitem(
            batch_size=[5],
            X_shape=(16, 32),
            split_sections=[8, 20, 4],
            split_dim=2,
            item_idx=1,
            test_name=f"split_getitem_{dtype}",
            dtype=dtype,
        )

    def _test_split_getitem_output(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32),
        split_sections=(4, 8, 2, 2),
        split_dim=1,
        item_idx=0,
        test_name="split_getitem_output",
        dtype="float16",
    ):
        assert len(X_shape) == 2, f"expected X_shape to be 2 but got {X_shape}"
        target = detect_target()
        b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
        X = Tensor(
            shape=[b_dim, *X_shape],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )

        Y1 = ops.split()(X, split_sections, split_dim)
        Y = ops.getitem()(Y1, item_idx)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = get_random_torch_tensor(X_shape_pt, dtype=dtype)

            Y1_pt = torch.split(X_pt, split_sections, split_dim)
            Y_pt = Y1_pt[item_idx]
            y = torch.empty_like(Y_pt)
            module.run_with_tensors([X_pt], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32")],
            }
        )
    )
    def test_split_getitem_output(self, dtype):
        self._test_split_getitem_output(
            test_name="split_getitem_output",
            dtype=dtype,
        )
        self._test_split_getitem_output(
            batch_size=[10],
            X_shape=(16, 31),
            split_sections=[9, 19, 3],
            split_dim=2,
            item_idx=1,
            test_name="split_getitem_output",
            dtype=dtype,
        )

    def _test_split_multiple_getitems(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32),
        split_sections=(4, 4, 6, 2),
        split_dim=1,
        test_name="split_multiple_getitems",
        dtype="float16",
    ):
        assert len(X_shape) == 2, f"expected X_shape to be 2 but got {X_shape}"
        assert (
            len(split_sections) >= 2
        ), f"expected split_sections to have at least 2 values, but got {split_sections}"
        target = detect_target()
        b_dim = shape_utils.gen_int_var_min_max(batch_size, name="input_batch")
        X = Tensor(
            shape=[b_dim, *X_shape],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        X2_shape = list(X_shape)
        item_idx0 = 0
        item_idx1 = 1
        assert split_sections[item_idx0] == split_sections[item_idx1], (
            f"expected values of split_sections at {item_idx0} and {item_idx1} "
            "are equal, but got {split_sections[item_idx0]} and "
            "{split_sections[item_idx1]}"
        )
        X2_shape[split_dim - 1] = split_sections[item_idx0]
        X2 = Tensor(
            shape=[b_dim, *X2_shape],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )

        Y1 = ops.split()(X, split_sections, split_dim)
        Y2 = ops.getitem()(Y1, item_idx0)
        Y3 = ops.getitem()(Y1, item_idx1)
        Y4 = ops.elementwise(FuncEnum.ADD)(Y2, X2)
        Y5 = ops.elementwise(FuncEnum.ADD)(Y3, Y3)
        Y = ops.elementwise(FuncEnum.ADD)(Y4, Y5)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = get_random_torch_tensor(X_shape_pt, dtype=dtype)
            X2_shape_pt = (b, *X2_shape)
            X2_pt = get_random_torch_tensor(X2_shape_pt, dtype=dtype)

            Y1_pt = torch.split(X_pt, split_sections, split_dim)
            Y2_pt = Y1_pt[item_idx0]
            Y3_pt = Y1_pt[item_idx1]
            Y4_pt = Y2_pt + X2_pt
            Y5_pt = Y3_pt + Y3_pt
            Y_pt = Y4_pt + Y5_pt

            y = torch.empty_like(Y_pt)
            module.run_with_tensors({"input_0": X_pt, "input_2": X2_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float32")],
            }
        )
    )
    def test_split_mutiple_getitems(self, dtype):
        self._test_split_multiple_getitems(
            test_name=f"split_multiple_getitems_{dtype}",
            dtype=dtype,
        )
        self._test_split_multiple_getitems(
            batch_size=[10],
            X_shape=(16, 31),
            split_sections=[9, 9, 13],
            split_dim=2,
            test_name=f"split_multiple_getitems_{dtype}",
            dtype=dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
