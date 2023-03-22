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
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntVarTensor

from aitemplate.frontend import IntImm, IntVar, nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


class ReshapeTestCase(unittest.TestCase):
    def _infer_shape(self, x, shape):
        new_shape = list(shape)
        cur_shape = x
        unknown_idx = -1
        prod = 1
        for idx, v in enumerate(new_shape):
            if v == -1:
                # no multiple -1s
                assert unknown_idx == -1
                unknown_idx = idx
            else:
                prod *= v
        numel = 1
        for dim in cur_shape:
            numel *= dim

        if unknown_idx == -1:
            assert (
                numel == prod
            ), f"When there is no unknown index, we expect dim products to be equal, got current shape {numel=} != new shape {prod=}"
        else:
            # FIXME: note that this RuntimeError rules out some "valid" PyTorch
            # code like:
            # t = torch.arange(0).reshape(4, 0)
            # this is valid in PT but would trigger RuntimeError below
            # t.reshape(2, 2, -1)
            # We can fix it later.
            if prod <= 0:
                raise RuntimeError(f"cannot reshape tensor {x} with shape {shape}")
            assert numel % prod == 0
            new_shape[unknown_idx] = numel // prod
        return new_shape

    def _test_reshape(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32, 64),
        Y_shape=(-1, 16, 16, 128),
        test_name="reshape",
        input_type="float16",
    ):
        target = detect_target()
        # N, H, W, C
        X = Tensor(
            shape=[IntVar(values=list(batch_size), name="input_batch"), *X_shape],
            dtype=input_type,
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

        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            # C, H, W
            X_shape_pt = (X_shape[2], X_shape[0], X_shape[1])
            X_pt = get_random_torch_tensor(shape=(b, *X_shape_pt), dtype=input_type)
            OP_pt = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            Y1_pt = OP_pt(X_pt).permute([0, 2, 3, 1])
            Y2_pt = torch.reshape(Y1_pt, shape)  # reshape 1
            Y_pt = torch.reshape(Y2_pt, shape + [1])  # reshape 2

            x = X_pt.permute((0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt)
            module.run_with_tensors([x], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _test_reshape_single_op(
        self,
        X_shape=(16, 32, 64),
        Y_shape=(-1, 16, 16, 128),
        test_name="reshape",
        check_name_retention=False,
        input_type="float16",
    ):
        target = detect_target()
        X_shape = [dim if isinstance(dim, IntVar) else IntImm(dim) for dim in X_shape]
        Y_shape = [dim if isinstance(dim, IntVar) else IntImm(dim) for dim in Y_shape]
        X = Tensor(
            shape=X_shape,
            dtype=input_type,
            name="input_0",
            is_input=True,
        )

        OP = nn.Reshape()
        Y = OP(X, Y_shape)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", test_name)

        # yank shape inference from op internals to let pt know what the real runtime shape will be
        x_shapes = list(itertools.product(*[var._attrs["values"] for var in X_shape]))
        new_shapes = list(itertools.product(*[var._attrs["values"] for var in Y_shape]))
        if len(x_shapes) > len(new_shapes):
            assert len(new_shapes) == 1
            new_shapes = new_shapes * len(x_shapes)

        y_shapes = [
            self._infer_shape(x_shape, new_shape)
            for x_shape, new_shape in zip(x_shapes, new_shapes)
        ]

        for x_shape, y_shape in zip(x_shapes, y_shapes):
            X_pt = get_random_torch_tensor(x_shape, input_type)
            Y_pt = torch.reshape(X_pt, y_shape)
            y = torch.empty_like(Y_pt)
            module.run_with_tensors([X_pt], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        if check_name_retention:
            self.assertTrue(
                1
                == sum("input_batch" == dim._attrs["name"] for dim in Y._attrs["shape"])
            )

    def test_reshape(self):
        self._test_reshape(test_name="reshape0")
        self._test_reshape([4, 2], (4, 8, 8), (-1,), "reshape1")
        self._test_reshape([3, 1], (5, 4, 16), (-1, 8), "reshape2")
        self._test_reshape_single_op(
            X_shape=(IntVar(values=(1, 3), name="input_batch"), 16, 32, 64),
            Y_shape=(-1, 16, 16, 128),
            test_name="reshape3",
        )
        self._test_reshape_single_op(
            X_shape=(1, 16, 32, 64), Y_shape=[1, 64, 16, 32], test_name="reshape4"
        )
        self._test_reshape_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 0, 8),
            Y_shape=(0, 2, 4),
            test_name="reshape1",
        )
        self._test_reshape_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 1, 120),
            Y_shape=(5, 4, -1, 3, 2),
            test_name="reshape_name",
            check_name_retention=True,
        )
        self._test_reshape_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 1, 120),
            Y_shape=(5, 4, IntVar(values=(2, 4), name="input_batch"), 3, -1),
            test_name="reshape_name_unknown_static_dim",
            check_name_retention=True,
        )
        self._test_reshape_single_op(
            X_shape=(IntVar(values=(2, 4), name="input_batch"), 1, 120),
            Y_shape=(5, IntVar(values=(2, 4)), 3, 4, 2),
            test_name="reshape_name_no_unknown_dims",
            check_name_retention=True,
        )
        self._test_reshape_single_op(
            X_shape=(IntVar(values=(20, 40), name="input_batch"), 1, 12),
            Y_shape=(4, 2, IntVar(values=(2, 4)), 3, 5),
            test_name="reshape_unsqueeze_intvar_dim",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_reshape_float32(self):
        self._test_reshape_single_op(input_type="float32", test_name="reshape_float32")

    def _test_reshape_shape(self, in_shape, out_shape, target_shape):
        X = Tensor(
            shape=in_shape,
            name="input_0",
            is_input=True,
        )

        OP = nn.Reshape()
        Y = OP(X, target_shape)

        y_shape = Y.shape()
        self.assertEqual(len(y_shape), len(out_shape))
        for y, o in zip(y_shape, out_shape):
            self.assertEqual(y, o)

    def test_reshape_shape_symbolic(self):
        dummy_shape = Tensor(
            shape=[1, 2],
            name="dummy_shape",
            is_input=True,
        )
        var1 = IntVar(values=[2, 4], name="var1")
        tensor1 = IntVarTensor(var1)
        X_shape = [var1, IntImm(256)]

        intvar = [ops.size()(dummy_shape, idx) for idx in range(2)]

        target_shape = [intvar[1] * tensor1, IntImm(-1)]
        outdim0 = IntVar(values=[4, 8])
        outdim0._attrs["symbolic_value"] = var1._attrs["symbolic_value"] * 2
        answer_shape = [outdim0, IntImm(128)]
        self._test_reshape_shape(X_shape, answer_shape, target_shape)


if __name__ == "__main__":
    unittest.main()
