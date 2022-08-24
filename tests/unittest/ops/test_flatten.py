# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import itertools
import unittest

import torch
from aitemplate.compiler.base import IntImm, IntVar

from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class FlattenTestCase(unittest.TestCase):
    def _test_fp16_single_op(
        self,
        X_shape,
        start_dim=0,
        end_dim=-1,
        test_name="flatten",
        check_name_retention=False,
    ):
        target = detect_target()
        dynamic_dim_names = [
            dim._attrs["name"] for dim in X_shape if isinstance(dim, IntVar)
        ]
        dynamic_dim_name = dynamic_dim_names[0] if 0 < len(dynamic_dim_names) else None
        X_shape = [dim if isinstance(dim, IntVar) else IntImm(dim) for dim in X_shape]
        X = Tensor(
            shape=X_shape,
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        OP = nn.Flatten(start_dim, end_dim)
        Y = OP(X)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = gen_execution_module(Y, target, "./tmp", test_name)

        x_shape_values = [var._attrs["values"] for var in X_shape]
        x_shapes = itertools.product(*x_shape_values)

        for x_shape in x_shapes:
            X_pt = torch.randn(x_shape).cuda().half()
            Y_pt = torch.flatten(X_pt, start_dim, end_dim)
            y = torch.empty_like(Y_pt)
            in_x = X_pt.clone()
            module.RunWithTensors([in_x], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
            if check_name_retention and dynamic_dim_name is not None:
                self.assertTrue(
                    1
                    == sum(
                        dynamic_dim_name == dim._attrs["name"]
                        for dim in Y._attrs["shape"]
                    )
                )

    def test_flatten(self):
        self._test_fp16_single_op(
            X_shape=(IntVar(values=[1, 3]), 16, 32, 64), test_name="flatten0"
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=[2, 5]), 16, 32, 64),
            start_dim=0,
            end_dim=1,
            test_name="flatten1",
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=[2, 5]), 16, 32, 64),
            start_dim=0,
            end_dim=0,
            test_name="flatten2",
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=[3, 4]), 16, 32, 64),
            start_dim=1,
            end_dim=-2,
            test_name="flatten3",
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=[3, 4], name="input_batch"), 16, 32, 2, 64),
            start_dim=1,
            end_dim=-2,
            test_name="flatten_name",
            check_name_retention=True,
        )
        self._test_fp16_single_op(
            X_shape=(16, 32, IntVar(values=[3, 4], name="input_batch"), 2, 64),
            start_dim=1,
            end_dim=-1,
            test_name="flatten_dynamic_nonbatch",
        )
        self._test_fp16_single_op(
            X_shape=(32, 16, 4, IntVar(values=[3, 4], name="input_batch"), 16),
            start_dim=0,
            end_dim=2,
            test_name="flatten_dynamic_nonbatch_name",
            check_name_retention=True,
        )

        self._test_fp16_single_op(
            X_shape=(32, 16, 4, 3, 16),
            start_dim=0,
            end_dim=2,
            test_name="flatten_static_1",
        )

        self._test_fp16_single_op(
            X_shape=(32, 3, 16, 4, 16),
            start_dim=0,
            end_dim=-1,
            test_name="flatten_static_2",
        )


if __name__ == "__main__":
    unittest.main()
