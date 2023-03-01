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
import logging
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils import shape_utils
from aitemplate.utils.torch_utils import string_to_torch_dtype


_LOGGER = logging.getLogger(__name__)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ReduceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ReduceTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

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
        rtol=1e-2,
        atol=1e-2,
    ):
        torch.manual_seed(0)
        _LOGGER.info(
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

        _LOGGER.info("AITemplate output_shape: {}".format(y_shape))
        _LOGGER.info("AITemplate output_type: {}".format(y_dtype))

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        X_pt = get_random_torch_tensor(input_shape, input_type)
        dtype_pt = string_to_torch_dtype(output_type)
        if keepdim is None:
            Y_pt = torch_reduce_op(X_pt, dim, dtype=dtype_pt)
        else:
            Y_pt = torch_reduce_op(X_pt, dim, keepdim=keepdim, dtype=dtype_pt)

        y = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y])

        torch.testing.assert_close(y_shape, Y_pt.shape)
        self.assertEqual(string_to_torch_dtype(y_dtype), Y_pt.dtype)
        torch.testing.assert_close(Y_pt, y, atol=atol, rtol=rtol)
        self.test_count += 1

    def _run_reduce_sum(
        self,
        *,
        dim,
        input_shape,
        keepdim,
        input_type="float16",
        output_type=None,
        rtol=1e-2,
        atol=1e-2,
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
            rtol=rtol,
            atol=atol,
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
        # allocate workspace for the strided tensor_reduce kernel
        self._run_reduce_sum(
            dim=2,
            input_shape=[1, 1, 8, 128],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )

    def _run_reduce_mean(
        self,
        *,
        dim,
        input_shape,
        keepdim,
        input_type="float16",
        output_type=None,
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

    def _run_batched_reduce(
        self,
        *,
        test_name,
        reduce_op,
        torch_reduce_op,
        dim,
        batch_sizes,
        non_batch_shape,
        keepdim,
        input_type="float16",
        output_type=None,
    ):
        torch.manual_seed(0)
        _LOGGER.info(f"Test {batch_sizes=}, {non_batch_shape=}, {dim=}")
        target = detect_target()

        batch0_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_0")
        non_batch_dims = [IntImm(d) for d in non_batch_shape]
        input_tensor_shape = [batch0_dim] + non_batch_dims
        X = Tensor(
            shape=input_tensor_shape, dtype=input_type, name="input_0", is_input=True
        )

        if keepdim is None:
            op = reduce_op(dim, dtype=output_type)
        else:
            op = reduce_op(dim, keepdim=keepdim, dtype=output_type)
        Y = op(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        for batch_size in batch_sizes:
            input_shape = [batch_size] + non_batch_shape
            X_pt = get_random_torch_tensor(input_shape, input_type)
            dtype_pt = (
                X_pt.dtype
                if output_type is None
                else string_to_torch_dtype(output_type)
            )
            if keepdim is None:
                Y_pt = torch_reduce_op(X_pt, dim, dtype=dtype_pt)
            else:
                Y_pt = torch_reduce_op(X_pt, dim, keepdim=keepdim, dtype=dtype_pt)

            y = torch.empty_like(Y_pt)
            module.run_with_tensors([X_pt], [y])

            torch.testing.assert_close(Y_pt, y, atol=1e-2, rtol=1e-2)
            self.test_count += 1

    def _run_batched_reduce_sum(
        self,
        *,
        dim,
        batch_sizes,
        non_batch_shape,
        keepdim,
        input_type="float16",
        output_type=None,
    ):
        self._run_batched_reduce(
            test_name="reduce_sum_batched",
            reduce_op=ops.reduce_sum,
            torch_reduce_op=torch.sum,
            dim=dim,
            batch_sizes=batch_sizes,
            non_batch_shape=non_batch_shape,
            keepdim=keepdim,
            input_type=input_type,
            output_type=output_type,
        )

    def test_batched_reduce_sum(self):
        self._run_batched_reduce_sum(
            dim=1,
            batch_sizes=[10, 2048],
            non_batch_shape=[2, 1944],
            keepdim=True,
            input_type="float16",
            output_type=None,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_reduce_sum_float32(self):
        # reduce_smallaxis
        self._run_reduce_sum(
            dim=1,
            input_shape=[1, 4],
            keepdim=True,
            input_type="float32",
            output_type=None,
            rtol=1.3e-6,
            atol=1e-5,
        )
        self._run_reduce_sum(
            dim=1,
            input_shape=[1, 4],
            keepdim=True,
            input_type="float16",
            output_type="float32",
            rtol=1.3e-6,
            atol=1e-5,
        )
        # reduce_3d
        self._run_reduce_sum(
            dim=-2,
            input_shape=[3, 2048, 4],
            keepdim=False,
            input_type="float32",
            output_type=None,
            rtol=4e-6,
            atol=2e-5,
        )
        self._run_reduce_sum(
            dim=1,
            input_shape=[11, 4096, 2],
            keepdim=True,
            input_type="float16",
            output_type="float32",
            rtol=1.3e-6,
            atol=1e-5,
        )
        # reduce (common) 2d
        self._run_reduce_sum(
            dim=-1,
            input_shape=[1270, 1223],
            keepdim=False,
            input_type="float32",
            output_type=None,
            rtol=1.3e-6,
            atol=1e-5,
        )
        self._run_reduce_sum(
            dim=0,
            input_shape=[1231, 1234],
            keepdim=True,
            input_type="float16",
            output_type="float32",
            rtol=1.3e-6,
            atol=1e-5,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_reduce_sum_bfloat16(self):
        # reduce_smallaxis
        self._run_reduce_sum(
            dim=1,
            input_shape=[1, 4],
            keepdim=True,
            input_type="bfloat16",
            output_type=None,
            rtol=1e-1,
            atol=1e-1,
        )
        # reduce_3d
        self._run_reduce_sum(
            dim=-2,
            input_shape=[3, 2048, 4],
            keepdim=False,
            input_type="bfloat16",
            output_type=None,
            rtol=1e-1,
            atol=1e-1,
        )
        # reduce (common) 2d
        self._run_reduce_sum(
            dim=-1,
            input_shape=[1270, 1223],
            keepdim=False,
            input_type="bfloat16",
            output_type=None,
            rtol=1e-0,
            atol=1e-0,
        )


if __name__ == "__main__":
    unittest.main()
