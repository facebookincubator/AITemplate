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
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import logger


# @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
@unittest.skip("BMM + Softmax is disabled for now")
class BMMSoftmaxTestCase(unittest.TestCase):
    def _test_bmm_rcr_softmax(
        self, B=16, M=16, K=64, N=24, test_name="bmm_rcr_softmax"
    ):

        X = Tensor(shape=[B, M, K], dtype="float16", name="input_0", is_input=True)
        W = Tensor(shape=[B, N, K], dtype="float16", name="input_1", is_input=True)
        OP = ops.bmm_rcr_softmax()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        if int(target._arch) < 80:
            logger.warning(__file__, "Skip this test on SM75")
            return
        module = compile_model(Y, target, "./tmp", test_name)
        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()

        WT = torch.transpose(W_pt, 2, 1)
        Y_pt = torch.bmm(X_pt, WT)
        Y_pt = torch.softmax(Y_pt, dim=-1)

        y = torch.empty([B, M, N]).cuda().half()
        module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
        eps = 1e-1
        self.assertTrue(torch.allclose(Y_pt, y, atol=eps, rtol=eps))

    def test_bmm_softmax(self):
        self._test_bmm_rcr_softmax()


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
