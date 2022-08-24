# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.testing.test_utils import get_random_torch_tensor


class SliceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceTestCase, self).__init__(*args, **kwargs)

    def _run_dynamic_slice(
        self, *, input_shape, start_indices, end_indices, input_type="float16"
    ):
        logging.info(
            "Test with start_indices {}, end_indices {}".format(
                start_indices, end_indices
            )
        )

        slice_op = ops.dynamic_slice()
        # generate torch reference result
        X_pt = get_random_torch_tensor(input_shape, input_type)
        slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
        Y_pt = X_pt[slice_indices]

        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)
        Y = slice_op(X, start_indices=start_indices, end_indices=end_indices)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate output_0 shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        module = gen_execution_module(Y, target, "./tmp", "dynamic_slice")

        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors([X_pt], [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _run_batch_dynamic_slice(
        self,
        *,
        batch_sizes,
        input_shape,
        start_indices,
        end_indices,
        input_type="float16",
    ):
        logging.info(
            "Batch test with batch_sizes {}, start_indices {}, end_indices {}".format(
                batch_sizes, start_indices, end_indices
            )
        )

        slice_op = ops.dynamic_slice()

        target = detect_target()
        X = Tensor(
            shape=[
                IntVar(values=batch_sizes, name="input_batch_0"),
                *input_shape,
            ],
            dtype=input_type,
            name="input_0",
            is_input=True,
        )
        Y = slice_op(X, start_indices=start_indices, end_indices=end_indices)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = gen_execution_module(Y, target, "./tmp", "dynamic_slice_batched")

        for batch in batch_sizes:
            logging.info("checking batch: {}".format(batch))

            # generate torch reference result
            X_pt = get_random_torch_tensor([batch, *input_shape], input_type)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            y_pt = Y_pt.cpu().numpy()

            y = torch.empty(y_pt.shape).cuda().half()
            module.RunWithTensors([X_pt], [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_dynamic_slice(self):
        self._run_dynamic_slice(input_shape=[1], start_indices=[0], end_indices=[1])
        self._run_dynamic_slice(input_shape=[2], start_indices=[0], end_indices=[-1])
        self._run_dynamic_slice(
            input_shape=[2, 3], start_indices=[0, 0], end_indices=[2, 2]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 5], start_indices=[0, 0, 0], end_indices=[2, 2, -1]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4], start_indices=[1, 0, 1], end_indices=[2, 2, 4]
        )
        self._run_dynamic_slice(
            input_shape=[2, 0, 4], start_indices=[0, 1, 0], end_indices=[1, 3, 4]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4], start_indices=[0, 1, 0], end_indices=[1, 3, 4]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4], start_indices=[0, 0, 0], end_indices=[1, 3, 4]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4], start_indices=[0, 1, 0], end_indices=[1, 3, -1]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4], start_indices=[0, 1, 1], end_indices=[-11, 3, 2]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4], start_indices=[0, -3, -4], end_indices=[9, -1, 2]
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4], start_indices=[4, 0, 1], end_indices=[1, 1, 2]
        )
        self._run_dynamic_slice(
            input_shape=[2048, 256, 64],
            start_indices=[256, 32, 0],
            end_indices=[1024, 193, 65],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 5], start_indices=[None, 0, 0], end_indices=[2, None, -1]
        )

    def test_batch_dynamic_slice(self):
        self._run_batch_dynamic_slice(
            batch_sizes=[1, 1],
            input_shape=[1],
            start_indices=[0, 0],
            end_indices=[1, 1],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 6, 7],
            input_shape=[2, 3, 4],
            start_indices=[2, 1, 0, 1],
            end_indices=[5, 2, 2, 4],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 6, 7],
            input_shape=[2, 3, 4],
            start_indices=[2, 1, 0, 1],
            end_indices=[-1, 2, -1, 4],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 6, 7],
            input_shape=[2, 3, 4],
            start_indices=[-5, 1, 0, 1],
            end_indices=[123, 2, -1, 4],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[7, 6, 5],
            input_shape=[128, 57, 74],
            start_indices=[1, 15, 32, 0],
            end_indices=[4, 73, 54, 65],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 6, 0],
            input_shape=[2, 3, 4],
            start_indices=[2, 1, 0, 1],
            end_indices=[-1, 2, -1, 4],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 3, 9],
            input_shape=[2, 0, 4],
            start_indices=[2, 1, 0, 1],
            end_indices=[-1, 2, -1, 4],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 3, 9],
            input_shape=[2, 4, 3],
            start_indices=[2, 1, 0, -1],
            end_indices=[-1, 2, -1, 0],
        )
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 3, 9],
            input_shape=[2, 4, 3],
            start_indices=[None, 1, None, -1],
            end_indices=[None, None, -1, 0],
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
