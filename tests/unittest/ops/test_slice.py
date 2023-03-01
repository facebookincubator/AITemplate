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
from aitemplate.compiler.base import IntVarTensor
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils import shape_utils


class DynamicSliceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DynamicSliceTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _run_dynamic_slice(
        self,
        *,
        input_shape,
        start_indices,
        end_indices,
        input_type="float16",
    ):
        logging.info(
            "Test with start_indices {}, end_indices {}".format(
                start_indices, end_indices
            )
        )

        slice_op = ops.dynamic_slice()
        # generate torch reference result
        X_pt = get_random_torch_tensor(input_shape, input_type)
        slice_indices = [
            slice(
                shape_utils.convert_IntVar_to_int(i) if i is not None else i,
                shape_utils.convert_IntVar_to_int(j) if j is not None else j,
            )
            for i, j in zip(start_indices, end_indices)
        ]
        Y_pt = X_pt[slice_indices]

        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)
        Y = slice_op(X, start_indices=start_indices, end_indices=end_indices)

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info("AITemplate output_0 shape: {}".format(y_shape))
        np.testing.assert_equal(y_shape, Y_pt.size())

        module = compile_model(Y, target, "./tmp", f"dynamic_slice_{self.test_count}")

        y_ait = torch.empty_like(Y_pt)
        module.run_with_tensors([X_pt], [y_ait])
        self.assertTrue(torch.allclose(Y_pt, y_ait, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_dynamic_slice(self):
        self._run_dynamic_slice(
            input_shape=[1],
            start_indices=[0],
            end_indices=[1],
        )
        self._run_dynamic_slice(
            input_shape=[2],
            start_indices=[0],
            end_indices=[-1],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3],
            start_indices=[0, 0],
            end_indices=[2, 2],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 5],
            start_indices=[0, 0, 0],
            end_indices=[2, 2, -1],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4],
            start_indices=[1, 0, 1],
            end_indices=[2, 2, 4],
        )
        self._run_dynamic_slice(
            input_shape=[2, 0, 4],
            start_indices=[0, 1, 0],
            end_indices=[1, 3, 4],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4],
            start_indices=[0, 1, 0],
            end_indices=[1, 3, 4],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4],
            start_indices=[0, 0, 0],
            end_indices=[1, 3, 4],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4],
            start_indices=[0, 1, 0],
            end_indices=[1, 3, -1],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4],
            start_indices=[0, 1, 1],
            end_indices=[-11, 3, 2],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4],
            start_indices=[0, -3, -4],
            end_indices=[9, -1, 2],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 4],
            start_indices=[4, 0, 1],
            end_indices=[1, 1, 2],
        )
        self._run_dynamic_slice(
            input_shape=[2048, 256, 64],
            start_indices=[256, 32, 0],
            end_indices=[1024, 193, 65],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3, 5],
            start_indices=[None, 0, 0],
            end_indices=[2, None, -1],
        )
        self._run_dynamic_slice(
            input_shape=[2, 3],
            start_indices=[IntVar([1, 1]), IntImm(1)],
            end_indices=[IntVarTensor(IntImm(2)), IntVarTensor(IntImm(2))],
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_dynamic_slice_float32(self):
        self._run_dynamic_slice(
            input_shape=[2, 3, 5],
            start_indices=[None, 0, 0],
            end_indices=[2, None, -1],
            input_type="float32",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_dynamic_slice_bfloat16(self):
        self._run_dynamic_slice(
            input_shape=[2, 3, 5],
            start_indices=[None, 0, 0],
            end_indices=[2, None, -1],
            input_type="bfloat16",
        )


class DynamicSliceBatchedTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DynamicSliceBatchedTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

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

        module = compile_model(
            Y, target, "./tmp", f"dynamic_slice_batched_{self.test_count}"
        )

        for batch in batch_sizes:
            logging.info("checking batch: {}".format(batch))

            # generate torch reference result
            X_pt = get_random_torch_tensor([batch, *input_shape], input_type)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            y_ait = torch.empty_like(Y_pt)
            module.run_with_tensors([X_pt], [y_ait])
            self.assertTrue(torch.allclose(Y_pt, y_ait, atol=1e-2, rtol=1e-2))
        self.test_count += 1

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

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_batch_dynamic_slice_float32(self):
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 3, 9],
            input_shape=[2, 4, 3],
            start_indices=[None, 1, None, -1],
            end_indices=[None, None, -1, 0],
            input_type="float32",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_batch_dynamic_slice_bfloat16(self):
        self._run_batch_dynamic_slice(
            batch_sizes=[5, 3, 9],
            input_shape=[2, 4, 3],
            start_indices=[None, 1, None, -1],
            end_indices=[None, None, -1, 0],
            input_type="bfloat16",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
