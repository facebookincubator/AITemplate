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
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class VectorNormTestCase(unittest.TestCase):
    def _run_vector_norm(
        self,
        *,
        test_name,
        dim,
        input_shape,
        ord_kind,
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

        op = ops.vector_norm(
            ord_kind=ord_kind, dim=dim, keepdim=keepdim, dtype=output_type
        )
        if copy_op:
            op = ops.vector_norm(**op._get_op_attributes())
        Y = op(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        y_dtype = Y._attrs["dtype"]

        logging.info("AITemplate output_shape: {}".format(y_shape))
        logging.info("AITemplate output_type: {}".format(y_dtype))

        module = compile_model(Y, target, "./tmp", test_name)
        X_pt = get_random_torch_tensor(input_shape, input_type)
        output_dtype_pt = (
            string_to_torch_dtype(output_type)
            if output_type is not None
            else string_to_torch_dtype(input_type)
        )
        Y_pt = torch.linalg.vector_norm(
            X_pt, ord=ord_kind, dim=dim, keepdim=keepdim, dtype=output_dtype_pt
        )
        y = torch.empty(y_shape, dtype=output_dtype_pt).cuda()
        module.run_with_tensors([X_pt], [y])

        self.assertEqual(y_shape, list(Y_pt.shape))
        self.assertEqual(string_to_torch_dtype(y_dtype), Y_pt.dtype)
        torch.testing.assert_close(Y_pt, y, atol=atol, rtol=rtol)

    def _run_l2_norm(
        self,
        *,
        dim,
        input_shape,
        keepdim,
        input_type="float16",
        output_type=None,
        atol=1e-2,
        rtol=1e-2,
    ):
        self._run_vector_norm(
            test_name="l2_norm",
            ord_kind=2,
            dim=dim,
            input_shape=input_shape,
            keepdim=keepdim,
            input_type=input_type,
            output_type=output_type,
            atol=atol,
            rtol=rtol,
        )
        self._run_vector_norm(
            test_name="l2_norm_copy_op",
            ord_kind=2,
            dim=dim,
            input_shape=input_shape,
            keepdim=keepdim,
            input_type=input_type,
            output_type=output_type,
            copy_op=True,
            atol=atol,
            rtol=rtol,
        )

    def test_l2_norm(self):
        self._run_l2_norm(dim=0, input_shape=[1], keepdim=True)
        self._run_l2_norm(dim=-1, input_shape=[3, 2, 2048], keepdim=False)
        self._run_l2_norm(dim=1, input_shape=[3, 1234, 4], keepdim=True)
        self._run_l2_norm(dim=1, input_shape=[5, 60, 34, 4], keepdim=False)
        self._run_l2_norm(dim=0, input_shape=[5, 60, 34, 4], keepdim=False)
        self._run_l2_norm(dim=2, input_shape=[5, 1, 34, 4], keepdim=False)
        self._run_l2_norm(dim=-1, input_shape=[4, 1230, 1237], keepdim=True)
        self._run_l2_norm(dim=-1, input_shape=[1, 1000000, 6], keepdim=True)

    def test_l2_norm_fp32(self):
        self._run_l2_norm(
            dim=0,
            input_shape=[1],
            keepdim=True,
            input_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=-1,
            input_shape=[3, 2, 2048],
            keepdim=False,
            input_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=1,
            input_shape=[3, 1234, 4],
            keepdim=True,
            input_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=1,
            input_shape=[5, 60, 34, 4],
            keepdim=False,
            input_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=0,
            input_shape=[5, 60, 34, 4],
            keepdim=False,
            input_type="float16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=2,
            input_shape=[5, 1, 34, 4],
            keepdim=False,
            input_type="float16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=-1,
            input_shape=[4, 1230, 1237],
            keepdim=True,
            input_type="float16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=-1,
            input_shape=[1, 1000000, 6],
            keepdim=True,
            input_type="float16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )

    def test_l2_norm_bf16(self):
        self._run_l2_norm(
            dim=0,
            input_shape=[1],
            keepdim=True,
            input_type="bfloat16",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=-1,
            input_shape=[3, 2, 2048],
            keepdim=False,
            input_type="bfloat16",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=1,
            input_shape=[3, 1234, 4],
            keepdim=True,
            input_type="bfloat16",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=1,
            input_shape=[5, 60, 34, 4],
            keepdim=False,
            input_type="bfloat16",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=0,
            input_shape=[5, 60, 34, 4],
            keepdim=False,
            input_type="bfloat16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=2,
            input_shape=[5, 1, 34, 4],
            keepdim=False,
            input_type="bfloat16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=-1,
            input_shape=[4, 1230, 1237],
            keepdim=True,
            input_type="bfloat16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )
        self._run_l2_norm(
            dim=-1,
            input_shape=[1, 1000000, 6],
            keepdim=True,
            input_type="bfloat16",
            output_type="float32",
            atol=1e-1,
            rtol=1e-1,
        )

    def _run_batched_vector_norm(
        self,
        *,
        test_name,
        dim,
        ord_kind,
        keepdim=False,
        input_type="float16",
        output_type=None,
    ):
        torch.manual_seed(0)
        logging.info(
            "Test batched_vector_norm with reduction_axes={dim}".format(dim=dim)
        )
        target = detect_target()

        M = 4
        N = 32
        X = Tensor(
            shape=[IntImm(M), IntVar(name="input_batch", values=[1, 2048]), IntImm(N)],
            dtype=input_type,
            name="input_0",
            is_input=True,
        )

        op = ops.vector_norm(
            ord_kind=ord_kind, dim=dim, keepdim=keepdim, dtype=output_type
        )
        Y = op(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_dtype = Y._attrs["dtype"]

        logging.info("AITemplate output_type: {}".format(y_dtype))

        output_dtype_pt = (
            string_to_torch_dtype(output_type)
            if output_type is not None
            else string_to_torch_dtype(input_type)
        )
        module = compile_model(Y, target, "./tmp", test_name)

        for B in [5, 128, 1024, 1237, 2002]:
            input_shape = [M, B, N]
            logging.info("Testing input_shape={}".format(input_shape))

            X_pt = get_random_torch_tensor(input_shape, input_type)
            Y_pt = torch.linalg.vector_norm(
                X_pt, ord=ord_kind, dim=dim, keepdim=keepdim, dtype=output_dtype_pt
            )
            y = torch.empty(Y_pt.shape, dtype=output_dtype_pt).cuda()
            module.run_with_tensors([X_pt], [y])

            self.assertEqual(string_to_torch_dtype(y_dtype), Y_pt.dtype)
            torch.testing.assert_close(Y_pt, y, atol=1e-2, rtol=1e-2)

    def _run_batched_l2_norm(
        self, *, dim, keepdim, input_type="float16", output_type=None
    ):
        self._run_batched_vector_norm(
            test_name="batched_l2_norm",
            ord_kind=2,
            dim=dim,
            keepdim=keepdim,
            input_type=input_type,
            output_type=output_type,
        )

    def test_batched_l2_norm(self):
        self._run_batched_l2_norm(dim=0, keepdim=True)
        self._run_batched_l2_norm(dim=1, keepdim=False)
        self._run_batched_l2_norm(dim=1, keepdim=True)
        self._run_batched_l2_norm(dim=2, keepdim=False)


if __name__ == "__main__":
    unittest.main()
