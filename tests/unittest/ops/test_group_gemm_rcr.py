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
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from parameterized import param, parameterized


_LOGGER = logging.getLogger(__name__)


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GroupGEMMRcrTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            param(False, "group_gemm_rcr_run_once", "float16"),
            param(True, "group_gemm_rcr_run_twice", "float16"),
            param(False, "group_gemm_rcr_run_once_fp32", "float32"),
            param(False, "group_gemm_rcr_run_once_bf16", "bfloat16"),
        ]
    )
    def test_group_gemm_rcr(self, run_twice: bool, test_name: str, dtype: str):
        M = 256
        K1 = 128
        N1 = 60
        K2 = 192
        N2 = 64
        target = detect_target()
        if int(target._arch) < 80:
            _LOGGER.warning("Group Gemm need SM80 HW")
            return
        X1 = Tensor(shape=[M, K1], dtype=dtype, name="x1", is_input=True)
        X2 = Tensor(shape=[M, K2], dtype=dtype, name="x2", is_input=True)
        W1 = Tensor(shape=[N1, K1], dtype=dtype, name="w1", is_input=True)
        W2 = Tensor(shape=[N2, K2], dtype=dtype, name="w2", is_input=True)
        OP = ops.group_gemm_rcr()
        Y1, Y2 = OP(operand_groups=[[X1, W1], [X2, W2]])
        Y1._attrs["name"] = "y1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "y2"
        Y2._attrs["is_output"] = True

        graph_outputs = [Y1, Y2]
        if run_twice:
            # Run twice to exercise having different unique_workspace offsets
            Y3 = ops.group_gemm_rcr()(operand_groups=[[X1, W1]])[0]
            Y3._attrs["name"] = "y3"
            Y3._attrs["is_output"] = True
            graph_outputs.append(Y3)

        module = compile_model(graph_outputs, target, "./tmp", test_name)
        X1_pt = get_random_torch_tensor(shape=(M, K1), dtype=dtype)
        X2_pt = get_random_torch_tensor(shape=(M, K2), dtype=dtype)
        W1_pt = get_random_torch_tensor(shape=(N1, K1), dtype=dtype)
        W2_pt = get_random_torch_tensor(shape=(N2, K2), dtype=dtype)
        Y1_pt = torch.nn.functional.linear(X1_pt, W1_pt)
        Y2_pt = torch.nn.functional.linear(X2_pt, W2_pt)

        inputs = {
            "x1": X1_pt,
            "w1": W1_pt,
            "x2": X2_pt,
            "w2": W2_pt,
        }
        y1 = torch.empty_like(Y1_pt)
        y2 = torch.empty_like(Y2_pt)
        outputs = {"y1": y1, "y2": y2}
        if run_twice:
            outputs["y3"] = torch.empty_like(y1)

        module.run_with_tensors(inputs, outputs)
        torch.testing.assert_close(Y1_pt, y1, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(Y2_pt, y2, atol=1e-1, rtol=1e-1)
        if run_twice:
            torch.testing.assert_close(Y1_pt, outputs["y3"], atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    unittest.main()
