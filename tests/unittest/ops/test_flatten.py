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
import itertools
import unittest

import torch

from aitemplate.compiler import compile_model
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FlattenTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def _test_single_op(
        self,
        X_shape,
        start_dim=0,
        end_dim=-1,
        test_name="flatten",
        check_name_retention=False,
        dtype="float16",
    ):
        target = detect_target()
        dynamic_dim_names = [
            dim._attrs["name"] for dim in X_shape if isinstance(dim, IntVar)
        ]
        dynamic_dim_name = dynamic_dim_names[0] if 0 < len(dynamic_dim_names) else None
        X_shape = [dim if isinstance(dim, IntVar) else IntImm(dim) for dim in X_shape]
        X = Tensor(
            shape=X_shape,
            dtype=dtype,
            name="input_0",
            is_input=True,
        )

        OP = nn.Flatten(start_dim, end_dim)
        Y = OP(X)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        x_shape_values = [var._attrs["values"] for var in X_shape]
        x_shapes = itertools.product(*x_shape_values)

        for x_shape in x_shapes:
            X_pt = get_random_torch_tensor(x_shape, dtype=dtype)
            Y_pt = torch.flatten(X_pt, start_dim, end_dim)
            y = torch.empty_like(Y_pt)
            in_x = X_pt.clone()
            module.run_with_tensors([in_x], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
            if check_name_retention and dynamic_dim_name is not None:
                self.assertTrue(
                    1
                    == sum(
                        dynamic_dim_name == dim._attrs["name"]
                        for dim in Y._attrs["shape"]
                    )
                )

    def test_flatten_fp16(self):
        self._test_single_op(
            X_shape=(IntVar(values=[1, 3]), 16, 32, 64),
            test_name="flatten_fp16",
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(IntVar(values=[2, 5]), 16, 32, 64),
            start_dim=0,
            end_dim=1,
            test_name="flatten_fp16",
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(IntVar(values=[2, 5]), 16, 32, 64),
            start_dim=0,
            end_dim=0,
            test_name="flatten_fp16",
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(IntVar(values=[3, 4]), 16, 32, 64),
            start_dim=1,
            end_dim=-2,
            test_name="flatten_fp16",
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(IntVar(values=[3, 4], name="input_batch"), 16, 32, 2, 64),
            start_dim=1,
            end_dim=-2,
            test_name="flatten_fp16_name",
            check_name_retention=True,
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(16, 32, IntVar(values=[3, 4], name="input_batch"), 2, 64),
            start_dim=1,
            end_dim=-1,
            test_name="flatten_fp16_dynamic_nonbatch",
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(32, 16, 4, IntVar(values=[3, 4], name="input_batch"), 16),
            start_dim=0,
            end_dim=2,
            test_name="flatten_fp16_dynamic_nonbatch_name",
            check_name_retention=True,
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(32, 16, 4, 3, 16),
            start_dim=0,
            end_dim=2,
            test_name="flatten_fp16_static",
            dtype="float16",
        )
        self._test_single_op(
            X_shape=(32, 3, 16, 4, 16),
            start_dim=0,
            end_dim=-1,
            test_name="flatten_fp16_static",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_flatten_fp32(self):
        self._test_single_op(
            X_shape=(IntVar(values=[1, 3]), 16, 32, 64),
            test_name="flatten_fp32",
            dtype="float32",
        )
        self._test_single_op(
            X_shape=(IntVar(values=[3, 4], name="input_batch"), 16, 32, 2, 64),
            start_dim=1,
            end_dim=-2,
            test_name="flatten_fp32_name",
            check_name_retention=True,
            dtype="float32",
        )
        self._test_single_op(
            X_shape=(16, 32, IntVar(values=[3, 4], name="input_batch"), 2, 64),
            start_dim=1,
            end_dim=-1,
            test_name="flatten_fp32_dynamic_nonbatch",
            dtype="float32",
        )
        self._test_single_op(
            X_shape=(32, 16, 4, 3, 16),
            start_dim=0,
            end_dim=2,
            test_name="flatten_fp32_static",
            dtype="float32",
        )

    def _test_flatten_shape(self, in_shape, out_shape, start_dim, end_dim):
        X = Tensor(
            shape=in_shape,
            name="input_0",
            is_input=True,
        )

        OP = nn.Flatten(start_dim, end_dim)
        Y = OP(X)

        y_shape = Y.shape()
        self.assertEqual(len(y_shape), len(out_shape))
        for y, o in zip(y_shape, out_shape):
            self.assertEqual(y, o)

    def test_flatten_shape_imm(self):
        in_shape = [IntImm(17), IntImm(19), IntImm(23)]

        self._test_flatten_shape(in_shape, [IntImm(17 * 19 * 23)], 0, 2)
        self._test_flatten_shape(in_shape, [IntImm(17 * 19 * 23)], 0, -1)
        self._test_flatten_shape(in_shape, [IntImm(17 * 19), IntImm(23)], 0, 1)
        self._test_flatten_shape(in_shape, [IntImm(17), IntImm(19 * 23)], 1, 2)
        self._test_flatten_shape(in_shape, [IntImm(17), IntImm(19 * 23)], 1, -1)

    def test_flatten_shape_var(self):
        var1 = IntVar(values=[1, 2], name="var1")
        var2 = IntVar(values=[3, 5], name="var2")
        var3 = IntVar(values=[7, 11], name="var3")
        sym1 = var1._attrs["symbolic_value"]
        sym2 = var2._attrs["symbolic_value"]
        sym3 = var3._attrs["symbolic_value"]
        in_shape = [var1, var2, var3]

        ovar1 = IntVar(values=[21, 110], symbolic_value=sym1 * sym2 * sym3)
        self._test_flatten_shape(in_shape, [ovar1], 0, 2)
        self._test_flatten_shape(in_shape, [ovar1], 0, -1)
        ovar1 = IntVar(values=[3, 10], symbolic_value=sym1 * sym2)
        self._test_flatten_shape(in_shape, [ovar1, var3], 0, 1)
        ovar1 = IntVar(values=[21, 55], symbolic_value=sym2 * sym3)
        self._test_flatten_shape(in_shape, [var1, ovar1], 1, 2)
        self._test_flatten_shape(in_shape, [var1, ovar1], 1, -1)

    def test_flatten_shape_mix(self):
        var1 = IntVar(values=[1, 2], name="var1")
        var2 = IntVar(values=[3, 5], name="var2")
        var3 = IntVar(values=[7, 11], name="var3")
        imm1 = IntImm(17)
        imm2 = IntImm(19)
        sym1 = var1._attrs["symbolic_value"]
        sym2 = var2._attrs["symbolic_value"]
        sym3 = var3._attrs["symbolic_value"]
        in_shape = [var1, imm1, var2, var3, imm2]

        ovar1 = IntVar(values=[51, 170])
        ovar1._attrs["symbolic_value"] = sym1 * 17 * sym2
        self._test_flatten_shape(in_shape, [ovar1, var3, imm2], 0, 2)
        ovar1 = IntVar(values=[6783, 35530], symbolic_value=323 * sym1 * sym2 * sym3)
        self._test_flatten_shape(in_shape, [ovar1], 0, -1)
        ovar1 = IntVar(values=[357, 935], symbolic_value=17 * sym2 * sym3)
        self._test_flatten_shape(in_shape, [var1, ovar1, imm2], 1, 3)
        ovar1 = IntVar(values=[6783, 17765], symbolic_value=323 * sym2 * sym3)
        self._test_flatten_shape(in_shape, [var1, ovar1], 1, -1)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
