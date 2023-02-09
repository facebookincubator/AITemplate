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

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class VarTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(VarTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _run_var(
        self,
        *,
        dim,
        unbiased,
        input_shape,
        keepdim=False,
        input_type="float16",
        output_type=None,
        copy_op=False,
        atol=1e-2,
        rtol=1e-2,
    ):
        torch.manual_seed(0)
        logging.info(
            "Test input_shape={input_shape}, reduction_axes={dim}".format(
                input_shape=input_shape, dim=dim
            )
        )
        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)

        op = ops.var(dim=dim, unbiased=unbiased, keepdim=keepdim, dtype=output_type)
        if copy_op:
            op = ops.var(**op._get_op_attributes())
        Y = op(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        y_dtype = Y._attrs["dtype"]

        logging.info("AITemplate output_shape: {}".format(y_shape))
        logging.info("AITemplate output_type: {}".format(y_dtype))

        module = compile_model(Y, target, "./tmp", f"var_{self.test_count}")
        X_pt = get_random_torch_tensor(input_shape, input_type)
        if output_type is None:
            torch_dtype = None
        else:
            torch_dtype = string_to_torch_dtype(output_type)
        Y_pt = torch.var(
            X_pt.to(dtype=torch_dtype), dim=dim, unbiased=unbiased, keepdim=keepdim
        )

        y = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y])

        np.testing.assert_equal(y_shape, Y_pt.size())
        np.testing.assert_equal(string_to_torch_dtype(y_dtype), Y_pt.dtype)
        self.assertTrue(torch.allclose(Y_pt, y, atol=atol, rtol=rtol, equal_nan=True))
        self.test_count += 1

    def test_var_float16(self):
        self._run_var(dim=-1, unbiased=True, input_shape=[1, 1], keepdim=False)
        self._run_var(dim=-1, unbiased=False, input_shape=[1, 1], keepdim=False)
        self._run_var(dim=-1, unbiased=True, input_shape=[1, 5], keepdim=False)
        self._run_var(dim=-1, unbiased=False, input_shape=[2, 8], keepdim=False)
        self._run_var(dim=-1, unbiased=False, input_shape=[3, 2, 2050], keepdim=False)
        self._run_var(dim=-1, unbiased=True, input_shape=[3, 2, 2050], keepdim=False)
        self._run_var(dim=1, unbiased=True, input_shape=[3, 2050, 2], keepdim=True)
        self._run_var(dim=0, unbiased=True, input_shape=[3001, 4, 2], keepdim=True)
        self._run_var(dim=-1, unbiased=True, input_shape=[1, 1000000, 6], keepdim=False)
        self._run_var(
            dim=0, unbiased=True, input_shape=[3001, 4, 2], keepdim=True, copy_op=True
        )
        self._run_var(
            dim=-1,
            unbiased=True,
            input_shape=[1, 1000000, 6],
            keepdim=False,
            copy_op=True,
        )

    def _run_batched_var(
        self, *, dim, unbiased, keepdim=False, input_type="float16", output_type=None
    ):
        torch.manual_seed(0)
        logging.info("Test batched_var with reduction_axes={dim}".format(dim=dim))
        target = detect_target()

        M = 4
        N = 32
        X = Tensor(
            shape=[IntImm(M), IntVar(name="input_batch", values=[1, 2048]), IntImm(N)],
            dtype=input_type,
            name="input_0",
            is_input=True,
        )

        op = ops.var(dim=dim, unbiased=unbiased, keepdim=keepdim, dtype=output_type)
        Y = op(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_dtype = Y._attrs["dtype"]

        logging.info("AITemplate output_type: {}".format(y_dtype))

        test_name = "batched_var"
        module = compile_model(Y, target, "./tmp", test_name)

        for B in [5, 128, 1024, 1237, 2002]:
            input_shape = [M, B, N]
            logging.info("Testing input_shape={}".format(input_shape))

            X_pt = get_random_torch_tensor(input_shape, input_type)
            Y_pt = torch.var(X_pt, dim=dim, unbiased=unbiased, keepdim=keepdim)

            y = torch.empty_like(Y_pt)
            module.run_with_tensors([X_pt], [y])

            np.testing.assert_equal(string_to_torch_dtype(y_dtype), Y_pt.dtype)
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_batched_var(self):
        self._run_batched_var(dim=0, unbiased=False, keepdim=True)
        self._run_batched_var(dim=1, unbiased=True, keepdim=False)
        self._run_batched_var(dim=1, unbiased=False, keepdim=True)
        self._run_batched_var(dim=2, unbiased=True, keepdim=False)

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_var_float32(self):
        self._run_var(
            dim=-1,
            unbiased=False,
            input_shape=[2, 8],
            keepdim=False,
            input_type="float32",
            atol=1.3e-6,
            rtol=1e-5,
        )
        self._run_var(
            dim=-1,
            unbiased=False,
            input_shape=[2, 8],
            keepdim=False,
            input_type="float16",
            output_type="float32",
            atol=1.3e-6,
            rtol=1e-5,
        )
        self._run_var(
            dim=-1,
            unbiased=False,
            input_shape=[3, 2, 2050],
            keepdim=False,
            input_type="float32",
            atol=1.3e-6,
            rtol=1e-5,
        )
        self._run_var(
            dim=-1,
            unbiased=False,
            input_shape=[3, 2, 2050],
            keepdim=False,
            input_type="float16",
            output_type="float32",
            atol=1.3e-6,
            rtol=1e-5,
        )
        self._run_var(
            dim=1,
            unbiased=True,
            input_shape=[1025, 2047],
            keepdim=True,
            input_type="float32",
            atol=1.3e-6,
            rtol=1e-5,
        )
        self._run_var(
            dim=1,
            unbiased=True,
            input_shape=[1025, 2047],
            keepdim=True,
            input_type="float16",
            output_type="float32",
            atol=1.3e-6,
            rtol=1e-5,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "bf16 not supported in ROCm")
    def test_var_bfloat16(self):
        self._run_var(
            dim=-1,
            unbiased=False,
            input_shape=[2, 8],
            keepdim=False,
            input_type="bfloat16",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_var(
            dim=-1,
            unbiased=False,
            input_shape=[3, 2, 2050],
            keepdim=False,
            input_type="bfloat16",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_var(
            dim=1,
            unbiased=True,
            input_shape=[1025, 2047],
            keepdim=True,
            input_type="bfloat16",
            atol=1e-1,
            rtol=1e-1,
        )


if __name__ == "__main__":
    unittest.main()
