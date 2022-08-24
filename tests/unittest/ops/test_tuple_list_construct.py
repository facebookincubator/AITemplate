# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from aitemplate.compiler import ops

from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class TupleConstructTestCase(unittest.TestCase):
    def _test_tuple_construct(
        self,
        batch_size=(1, 3),
        X_shape=(16, 32, 64),
        test_op=ops.tuple_construct,
        test_name="tuple",
    ):
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), *X_shape],
            dtype="float16",
            name="input_0",
            is_input=True,
        )

        X1 = ops.reshape()(X, [-1, X_shape[-1]])
        X2 = ops.flatten()(X)
        X3 = ops.unsqueeze(1)(X2)
        T = test_op()(X1, X2, X3)
        Y1 = ops.getitem()(T, 0)
        Y2 = ops.getitem()(T, 1)
        Y3 = ops.getitem()(T, 2)

        Y1._attrs["name"] = "output_0"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "output_1"
        Y2._attrs["is_output"] = True
        Y3._attrs["name"] = "output_2"
        Y3._attrs["is_output"] = True

        module = gen_execution_module([Y1, Y2, Y3], target, "./tmp", test_name)

        for b in batch_size:
            X_shape_pt = (b, *X_shape)
            X_pt = torch.randn(X_shape_pt).cuda().half()
            Y1_pt = X_pt.reshape(-1, X_shape_pt[-1])
            Y2_pt = X_pt.flatten()
            Y3_pt = Y2_pt.unsqueeze(1)

            outputs = [
                torch.empty(Y1_pt.size()).cuda().half(),
                torch.empty(Y2_pt.size()).cuda().half(),
                torch.empty(Y3_pt.size()).cuda().half(),
            ]
            module.RunWithTensors([X_pt], outputs)

            self.assertTrue(torch.allclose(Y1_pt, outputs[0], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(Y2_pt, outputs[1], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(Y3_pt, outputs[2], atol=1e-2, rtol=1e-2))

    def test_tuple_construct(self):
        self._test_tuple_construct(test_op=ops.tuple_construct, test_name="tuple_0")
        self._test_tuple_construct(test_op=ops.list_construct, test_name="list_0")


if __name__ == "__main__":
    unittest.main()
