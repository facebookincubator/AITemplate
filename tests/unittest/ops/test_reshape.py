# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import itertools
import unittest

import torch
from aitemplate.compiler.ops.common.view_ops import reshape

from aitemplate.frontend import IntImm, IntVar, nn, Tensor
from aitemplate.testing import detect_target, gen_execution_module


class ReshapeTestCase(unittest.TestCase):
    def _test_fp16(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32, 64),
        Y_shape=(-1, 16, 16, 128),
        test_name="reshape",
    ):
        target = detect_target()
        # N, H, W, C
        X = Tensor(
            shape=[IntVar(values=list(batch_size), name="input_batch"), *X_shape],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        shape = list(Y_shape)

        OP1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        OP2 = nn.Reshape()
        OP3 = nn.Reshape()

        Y1 = OP1(X)
        Y2 = OP2(Y1, shape)
        Y = OP3(Y2, shape + [1])

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = gen_execution_module(Y, target, "./tmp", test_name)

        for b in batch_size:
            # C, H, W
            X_shape_pt = (X_shape[2], X_shape[0], X_shape[1])
            X_pt = torch.randn(b, *X_shape_pt).cuda().half()
            OP_pt = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            Y1_pt = OP_pt(X_pt).permute([0, 2, 3, 1])
            Y2_pt = torch.reshape(Y1_pt, shape)  # reshape 1
            Y_pt = torch.reshape(Y2_pt, shape + [1])  # reshape 2

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty(Y_pt.size()).cuda().half()
            module.RunWithTensors([x], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _test_fp16_single_op(
        self,
        X_shape,
        Y_shape,
        test_name="reshape",
        check_name_retention=False,
    ):
        target = detect_target()
        X_shape = [dim if isinstance(dim, IntVar) else IntImm(dim) for dim in X_shape]
        Y_shape = [dim if isinstance(dim, IntVar) else IntImm(dim) for dim in Y_shape]
        X = Tensor(
            shape=X_shape,
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        OP = nn.Reshape()
        OP_backend = reshape()
        Y = OP(X, Y_shape)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = gen_execution_module(Y, target, "./tmp", test_name)

        # yank shape inference from op internals to let pt know what the real runtime shape will be
        x_shapes = list(itertools.product(*[var._attrs["values"] for var in X_shape]))
        new_shapes = list(itertools.product(*[var._attrs["values"] for var in Y_shape]))
        if len(x_shapes) > len(new_shapes):
            assert len(new_shapes) == 1
            new_shapes = new_shapes * len(x_shapes)
        y_shapes = [
            OP_backend._infer_shape(x_shape, new_shape)
            for x_shape, new_shape in zip(x_shapes, new_shapes)
        ]

        for x_shape, y_shape in zip(x_shapes, y_shapes):
            X_pt = torch.randn(x_shape).cuda().half()
            Y_pt = torch.reshape(X_pt, y_shape)
            y = torch.empty(Y_pt.size()).cuda().half()
            module.RunWithTensors([X_pt], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        if check_name_retention:
            self.assertTrue(
                1
                == sum("input_batch" == dim._attrs["name"] for dim in Y._attrs["shape"])
            )

    def test_reshape(self):
        self._test_fp16(test_name="reshape0")
        self._test_fp16([4, 2], (4, 8, 8), (-1,), "reshape1")
        self._test_fp16([3, 1], (5, 4, 16), (-1, 8), "reshape2")
        self._test_fp16_single_op(
            X_shape=(IntVar(values=(1, 3), name="input_batch"), 16, 32, 64),
            Y_shape=(-1, 16, 16, 128),
            test_name="reshape3",
        )
        self._test_fp16_single_op(
            X_shape=(1, 16, 32, 64), Y_shape=[1, 64, 16, 32], test_name="reshape4"
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 0, 8),
            Y_shape=(0, 2, 4),
            test_name="reshape1",
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 1, 120),
            Y_shape=(5, 4, -1, 3, 2),
            test_name="reshape_name",
            check_name_retention=True,
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 1, 120),
            Y_shape=(5, 4, IntVar(values=(2, 4)), 3, -1),
            test_name="reshape_name_unknown_static_dim",
            check_name_retention=True,
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 1, 120),
            Y_shape=(5, IntVar(values=(2, 4)), 3, 4, 2),
            test_name="reshape_name_no_unknown_dims",
            check_name_retention=True,
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 1, 120),
            Y_shape=(IntVar(values=(10, 20)), 4, 2, 3, -1),
            test_name="reshape_squeeze_intvar_dim",
        )
        self._test_fp16_single_op(
            X_shape=(IntVar(values=(20, 40), name="input_batch"), 1, 12),
            Y_shape=(4, 2, IntVar(values=(2, 4)), 3, 5),
            test_name="reshape_unsqueeze_intvar_dim",
        )


if __name__ == "__main__":
    unittest.main()
