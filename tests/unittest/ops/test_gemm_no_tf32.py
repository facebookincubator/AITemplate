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
from aitemplate.testing.test_utils import filter_test_cases_by_test_env


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
@unittest.skipIf(
    detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
    "Not supported by CUDA < SM80.",
)
class GEMMNoTF32TestCase(unittest.TestCase):
    def test_rrr_no_tf32(self):
        # Test accuracy with tf32 disabled
        # this test uses a smaller numerical tolerance level
        # than the others
        allow_tf32_bak = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            test_dtype = torch.float32
            test_dtype_str = "float32"
            A = torch.rand((64, 64), dtype=test_dtype).cuda()
            B = torch.rand((64, 64), dtype=test_dtype).cuda()
            result_cuda = torch.matmul(A, B)

            target = detect_target(no_tf32=True)  # Disable tf32 for accuracy
            A_ait = Tensor(
                shape=[64, 64], dtype=test_dtype_str, name="input_0", is_input=True
            )
            B_ait = Tensor(
                shape=[64, 64], dtype=test_dtype_str, name="input_1", is_input=True
            )
            OP = ops.gemm_rrr()
            Y = OP(A_ait, B_ait)
            Y._attrs["name"] = "output_0"
            Y._attrs["is_output"] = True
            module = compile_model(Y, target, "./tmp", "gemm_rrr_no_tf32")
            inputs = {
                "input_0": A.clone().detach().cuda(),
                "input_1": B.clone().detach().cuda(),
            }
            result_ait = torch.empty([64, 64], dtype=test_dtype, device="cuda")
            module.run_with_tensors(inputs, [result_ait])
            torch.testing.assert_close(result_cuda, result_ait)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32_bak


filter_test_cases_by_test_env(GEMMNoTF32TestCase)


if __name__ == "__main__":
    unittest.main()
