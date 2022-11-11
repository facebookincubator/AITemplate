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

from aitemplate.compiler import compile_model

from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target


class TensorTestCase(unittest.TestCase):
    def test_tensor_size(self):
        to_torch_dtype = {
            "bool": torch.bool,
            "int": torch.int32,
            "int32": torch.int32,
            "int64": torch.int64,
            "float16": torch.float16,
            "float": torch.float,
            "float32": torch.float,
        }
        for dtype, torch_dtype in to_torch_dtype.items():
            x = Tensor([3], dtype=dtype, is_input=True, is_output=True)
            x_pt = torch.randn(3).to(torch_dtype).cuda()

            expected_bytes = x_pt.numel() * x_pt.element_size()
            self.assertEqual(x.size_bytes(), expected_bytes)

            mod = compile_model(
                x, detect_target(), "./tmp", f"test_tensor_size_{dtype}"
            )

            out = torch.empty_like(x_pt)
            mod.run_with_tensors([x_pt], [out])
            self.assertTrue(torch.equal(out, x_pt))


if __name__ == "__main__":
    unittest.main()
