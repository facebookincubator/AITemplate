# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import random
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.testing.test_utils import get_random_torch_tensor


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GatherTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GatherTestCase, self).__init__(*args, **kwargs)

    def _run_gather_test(self, *, input_shape, gather_dim, dim_size, index_shape=None):
        logging.info(
            "Test with input_shape {}, gather_dim {}".format(input_shape, gather_dim)
        )

        input_type = "float16"
        index_type = "int64"
        if index_shape is None:
            index_shape = [
                random.randint(0, d - 1) if i != gather_dim else dim_size
                for (i, d) in enumerate(input_shape)
            ]
        logging.info("index_shape {}".format(index_shape))

        X = Tensor(shape=input_shape, dtype=input_type, name="X", is_input=True)
        Index = Tensor(shape=index_shape, dtype=index_type, name="Index", is_input=True)
        gather_op = ops.gather()
        Y = gather_op(X, gather_dim, Index)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        np.testing.assert_equal(y_shape, index_shape)

        target = detect_target()
        module = gen_execution_module(Y, target, "./tmp", "gather")

        X_pt = get_random_torch_tensor(input_shape, input_type)
        Index_pt = torch.randint(
            input_shape[gather_dim], size=index_shape, dtype=torch.int64
        ).cuda()
        Y_pt = torch.gather(X_pt, gather_dim, Index_pt)
        Y_np = Y_pt.cpu().numpy()
        np.testing.assert_equal(y_shape, Y_np.shape)

        Index_pt = Index_pt.to(torch.int64)
        inputs = {"X": X_pt, "Index": Index_pt}
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_gather(self):
        self._run_gather_test(input_shape=[2], gather_dim=0, dim_size=1)
        self._run_gather_test(input_shape=[2], gather_dim=0, dim_size=2)
        self._run_gather_test(input_shape=[2], gather_dim=0, dim_size=3)

        self._run_gather_test(input_shape=[3, 4, 5], gather_dim=2, dim_size=7)
        self._run_gather_test(input_shape=[3, 4, 5], gather_dim=1, dim_size=4)
        self._run_gather_test(
            input_shape=[3, 4, 5], gather_dim=0, dim_size=2, index_shape=[7, 1, 4]
        )
        self._run_gather_test(
            input_shape=[3, 4, 5], gather_dim=2, dim_size=7, index_shape=[0, 1, 2]
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
