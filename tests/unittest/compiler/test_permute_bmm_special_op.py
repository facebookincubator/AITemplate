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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import shape_utils


class FusePermuteBmmRRRN1Case(unittest.TestCase):
    def _test_permute_bmm_rrr_n1(self, B, M, K, testname, dtype="float16"):
        N = 1

        batch_dim = shape_utils.gen_int_var_min_max(B)
        X = Tensor(shape=[batch_dim, M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[batch_dim, N, K], dtype=dtype, name="input_1", is_input=True)

        WT = ops.permute021()(W)

        Y = ops.bmm_rrr()(X, WT)
        Y._attrs["name"] = "bmm_rrr_tensor"

        output = ops.elementwise(FuncEnum.COS)(Y)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        bmm_tensor = None
        for tensor in module.debug_sorted_graph:
            if tensor._attrs["name"] == "bmm_rrr_tensor":
                bmm_tensor = tensor
                break

        assert len(bmm_tensor._attrs["src_ops"]) == 1
        src_op = list(bmm_tensor._attrs["src_ops"])[0]
        assert src_op._attrs["op"] == "bmm_rcr_n1"

        for b in B:
            X_pt = get_random_torch_tensor([b, M, K], dtype)
            W_pt = get_random_torch_tensor([b, K, N], dtype)

            Y_pt = torch.cos(torch.bmm(X_pt, W_pt))
            w = W_pt.permute([0, 2, 1]).contiguous()

            # We currently only have row-major outputs.
            y = get_torch_empty_tensor([b, M, N], dtype)
            module.run_with_tensors({"input_0": X_pt, "input_1": w}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_permute_bmm_rrr_n1_fp16(self):
        self._test_permute_bmm_rrr_n1([1], 4, 8, "permute_bmm_rrr_n1_fp16")
        self._test_permute_bmm_rrr_n1([1, 3], 4, 8, "permute_bmm_rrr_n1_dynamic_fp16")

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_permute_bmm_rrr_n1_fp32(self):
        self._test_permute_bmm_rrr_n1(
            [1, 3], 4, 8, "permute_bmm_rrr_n1_dynamic_fp32", dtype="float32"
        )


if __name__ == "__main__":
    unittest.main()
