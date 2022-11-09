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

from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target, test_utils


class FusePermuteGemmTestCase(unittest.TestCase):
    def test_no_fusion_odd_alignment(self):
        x = Tensor([32, 51], is_input=True)
        w = Tensor([32, 51], is_input=True)
        y = ops.permute()(x, dims=[1, 0])
        z = ops.gemm_rrr()(w, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        module = compile_model(
            z, detect_target(), "./tmp", "test_no_fusion_odd_alignment"
        )
        self.assertTrue(test_utils.graph_has_op(module.debug_sorted_graph, "permute"))
        self.assertTrue(test_utils.graph_has_op(module.debug_sorted_graph, "gemm_rrr"))

    def test_gemm_rrr_to_rcr(self):
        x = Tensor([32, 52], is_input=True, name="x")
        w = Tensor([32, 52], is_input=True, name="w")
        y = ops.permute()(x, dims=[1, 0])
        z = ops.gemm_rrr()(w, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        module = compile_model(z, detect_target(), "./tmp", "test_gemm_rrr_to_rcr")
        self.assertFalse(test_utils.graph_has_op(module.debug_sorted_graph, "permute"))
        self.assertFalse(test_utils.graph_has_op(module.debug_sorted_graph, "gemm_rrr"))
        self.assertTrue(test_utils.graph_has_op(module.debug_sorted_graph, "gemm_rcr"))

        x_pt = torch.randn(32, 52).half().cuda()
        w_pt = torch.randn(32, 52).half().cuda()
        y_pt = x_pt.t()
        z_pt = torch.matmul(w_pt, y_pt)
        z_ait = torch.empty_like(z_pt)
        module.run_with_tensors({"x": x_pt, "w": w_pt}, {"z": z_ait})

        torch.testing.assert_close(z_ait, z_pt, atol=1e-1, rtol=1e-1)

    def test_gemm_rcr_to_rrr(self):
        x = Tensor([52, 32], is_input=True, name="x")
        w = Tensor([32, 52], is_input=True, name="w")
        y = ops.permute()(x, dims=[1, 0])
        z = ops.gemm_rcr()(w, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "z"

        module = compile_model(
            z,
            detect_target(),
            "./tmp",
            "test_gemm_rcr_to_rrr",
        )
        self.assertFalse(test_utils.graph_has_op(module.debug_sorted_graph, "permute"))
        self.assertFalse(test_utils.graph_has_op(module.debug_sorted_graph, "gemm_rcr"))
        self.assertTrue(test_utils.graph_has_op(module.debug_sorted_graph, "gemm_rrr"))

        x_pt = torch.randn(52, 32).half().cuda()
        w_pt = torch.randn(32, 52).half().cuda()
        z_pt = torch.matmul(w_pt, x_pt)
        z_ait = torch.empty_like(z_pt)
        module.run_with_tensors({"x": x_pt, "w": w_pt}, {"z": z_ait})

        torch.testing.assert_close(z_ait, z_pt, atol=1e-1, rtol=1e-1)
