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
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

logger = logging.getLogger(__name__)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class CUDAGraphTestCase(unittest.TestCase):
    def test_cuda_graph_multiple_runs(self):
        logger.info("testing cuda graph with multiple runs")
        X0_batch_dim = IntVar([1, 65], name="batch_size")
        X0_non_batch_shape = [1, 772]
        X0_non_batch_dims = [IntImm(d) for d in X0_non_batch_shape]
        X0_tensor_shape = [X0_batch_dim] + X0_non_batch_dims
        X0 = Tensor(shape=X0_tensor_shape, name="X0", is_input=True)
        X1_shape = [2, 772]
        X1 = Tensor(shape=X1_shape, name="X1", is_input=True)

        reduction_dim = 1
        Y0 = ops.elementwise(func_enum=FuncEnum.ADD)(X0, X1)
        Y = ops.reduce_sum(dim=reduction_dim)(Y0)

        # Set outputs
        Y._attrs["name"] = "Trueoutput_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        test_name = "cuda_graph_multiple_runes"
        module = compile_model(Y, target, "./tmp", test_name)

        run = 2
        repeat = 1
        for b_size in [1, 65]:
            logger.info(f"batch size = {b_size}")
            X0_shape = [b_size] + X0_non_batch_shape
            x0_pt = torch.randn(*X0_shape).cuda().half()
            x1_pt = torch.randn(*X1_shape).cuda().half()
            y0_pt = x0_pt + x1_pt
            y_pt = torch.sum(y0_pt, dim=reduction_dim)

            y = torch.empty(y_pt.size()).cuda().half()
            inputs = {"X0": x0_pt, "X1": x1_pt}
            module.run_with_tensors(inputs, [y])
            module.benchmark_with_tensors(
                inputs,
                [y],
                count=run,
                repeat=repeat,
                graph_mode=True,
            )
            y_pt = y_pt.cpu().numpy()
            np.testing.assert_allclose(y_pt, y.cpu().numpy(), atol=0.05, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
