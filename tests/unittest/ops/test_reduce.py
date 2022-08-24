# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.testing.test_utils import dtype_to_torch_dtype, get_random_torch_tensor


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ReduceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ReduceTestCase, self).__init__(*args, **kwargs)

    def _run_reduce(
        self,
        *,
        test_name,
        reduce_op,
        torch_reduce_op,
        dim,
        input_shape,
        keepdim,
        input_type="float16",
        output_type=None,
    ):
        torch.manual_seed(0)
        logging.info(
            "Test input_shape={input_shape}, reduction_axes={dim}".format(
                input_shape=input_shape, dim=dim
            )
        )
        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)

        if keepdim is None:
            op = reduce_op(dim, dtype=output_type)
        else:
            op = reduce_op(dim, keepdim=keepdim, dtype=output_type)
        Y = op(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        y_dtype = Y._attrs["dtype"]

        logging.info("AITemplate output_shape: {}".format(y_shape))
        logging.info("AITemplate output_type: {}".format(y_dtype))

        module = gen_execution_module(Y, target, "./tmp", test_name)
        X_pt = get_random_torch_tensor(input_shape, input_type)
        dtype_pt = dtype_to_torch_dtype(output_type)
        if keepdim is None:
            Y_pt = torch_reduce_op(X_pt, dim, dtype=dtype_pt)
        else:
            Y_pt = torch_reduce_op(X_pt, dim, keepdim=keepdim, dtype=dtype_pt)

        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors([X_pt], [y])
        y_pt = Y_pt.cpu().numpy()

        np.testing.assert_equal(y_shape, y_pt.shape)
        np.testing.assert_equal(dtype_to_torch_dtype(y_dtype), Y_pt.dtype)
        np.testing.assert_allclose(y_pt, y.cpu().numpy(), atol=1e-2, rtol=1e-2)

    def _run_reduce_sum(
        self, *, dim, input_shape, keepdim, input_type="float16", output_type=None
    ):
        self._run_reduce(
            test_name="reduce_sum",
            reduce_op=ops.reduce_sum,
            torch_reduce_op=torch.sum,
            dim=dim,
            input_shape=input_shape,
            keepdim=keepdim,
            input_type=input_type,
            output_type=output_type,
        )

    def test_reduce_sum(self):
        self._run_reduce_sum(
            dim=0, input_shape=[1], keepdim=True, input_type="float16", output_type=None
        )
        self._run_reduce_sum(
            dim=1,
            input_shape=[1, 4],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_sum(
            dim=0,
            input_shape=[1, 4],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_sum(
            dim=0,
            input_shape=[2, 4],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_sum(
            dim=0,
            input_shape=[1, 2, 1],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_sum(
            dim=1,
            input_shape=[1, 2, 1],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_sum(
            dim=2,
            input_shape=[5, 4, 3],
            keepdim=True,
            input_type="float16",
            output_type="float16",
        )

        self._run_reduce_sum(
            dim=0,
            input_shape=[4],
            keepdim=False,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_sum(
            dim=0,
            input_shape=[1, 4],
            keepdim=False,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_sum(
            dim=0,
            input_shape=[5, 4, 3],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_sum(
            dim=1,
            input_shape=[5, 4, 3],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_sum(
            dim=2,
            input_shape=[5, 4, 3],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_sum(
            dim=-1,
            input_shape=[1, 1000000, 6],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )

    def _run_reduce_mean(
        self, *, dim, input_shape, keepdim, input_type="float16", output_type=None
    ):
        self._run_reduce(
            test_name="reduce_mean",
            reduce_op=ops.reduce_mean,
            torch_reduce_op=torch.mean,
            dim=dim,
            input_shape=input_shape,
            keepdim=keepdim,
            input_type=input_type,
            output_type=output_type,
        )

    def test_reduce_mean(self):
        self._run_reduce_mean(
            dim=0, input_shape=[1], keepdim=True, input_type="float16", output_type=None
        )
        self._run_reduce_mean(
            dim=1,
            input_shape=[2, 3],
            keepdim=False,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_mean(
            dim=2,
            input_shape=[5, 7, 1234],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[3, 2, 2048],
            keepdim=True,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=0,
            input_shape=[2, 1],
            keepdim=False,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_mean(
            dim=0,
            input_shape=[4, 3],
            keepdim=False,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_mean(
            dim=1,
            input_shape=[2, 1, 3],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_mean(
            dim=1,
            input_shape=[3, 2057, 4],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_mean(
            dim=-2,
            input_shape=[3, 2048, 4],
            keepdim=False,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_mean(
            dim=1,
            input_shape=[0, 2048, 4],
            keepdim=False,
            input_type="float16",
            output_type=None,
        )
        self._run_reduce_mean(
            dim=2,
            input_shape=[0, 7, 1234],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=0,
            input_shape=[5, 7, 4],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=1,
            input_shape=[4, 5, 7, 4],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=2,
            input_shape=[4, 5, 7, 4],
            keepdim=True,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=3,
            input_shape=[4, 5, 7, 4],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[1, 1000000, 6],
            keepdim=True,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[1, 31],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[127, 63],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[1270, 63],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[4, 1280, 123],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[2, 22, 68],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )
        self._run_reduce_mean(
            dim=-1,
            input_shape=[1270, 1223],
            keepdim=False,
            input_type="float16",
            output_type="float16",
        )


if __name__ == "__main__":
    unittest.main()
