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
# BMM + Softmax + BMM
# (B, M, K) * (B, N, K) = (B, M, N) #RCR
# softmax on dim N (B, M, N)
# (B, M, N) * (B, N, O) = (B, M, O) #RRR
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


class SplitBMMTestCase(unittest.TestCase):
    def _test_split_reshape_bmm_permute(
        self, bs, nheads, seq_len, hidden_size, test_name, dtype="float16"
    ):
        target = detect_target()
        head_dim = hidden_size // nheads
        scale = head_dim**-0.5

        batch_dim = shape_utils.gen_int_var_min_max(bs, name="batch_size")
        input_shape = [3, batch_dim, nheads, seq_len, head_dim]
        X = Tensor(shape=input_shape, dtype=dtype, name="input_0", is_input=True)
        (Q, K, V) = ops.split()(X, 1, dim=0)

        OP = ops.bmm_softmax_bmm_permute(shape=(nheads,), scale=scale)
        Y = OP(
            (ops.reshape()(Q, [-1, seq_len, head_dim])),
            (ops.reshape()(K, [-1, seq_len, head_dim])),
            (ops.reshape()(V, [-1, seq_len, head_dim])),
        )

        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"split_bmm_softmax_bmm_{test_name}")
        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        # 1 input tensors + 1 output tensor
        self.assertEqual(len(sorted_graph), 2)
        # only bmm_softmax_bmm_permute left
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 1)

        for b in bs:
            input_shape = [3, b, nheads, seq_len, head_dim]
            x_pt = get_random_torch_tensor(input_shape, dtype)
            (q_pt, k_pt, v_pt) = torch.split(x_pt, 1, dim=0)
            q_pt = q_pt.reshape(-1, seq_len, head_dim)
            k_pt = k_pt.reshape(-1, seq_len, head_dim)
            v_pt = v_pt.reshape(-1, seq_len, head_dim)

            # [b, seq_len, head_dim] @ [b, head_dim, seq_len]
            attn = (q_pt @ k_pt.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            # [b, seq_len, seq_len] @ [b, seq_len, head_dim]
            y_l = attn @ v_pt
            y_r = y_l.reshape(b, nheads, seq_len, head_dim)
            y_pt = torch.permute(y_r, [0, 2, 1, 3])

            y = get_torch_empty_tensor([b, seq_len, nheads, head_dim], dtype)
            module.run_with_tensors([x_pt], [y])
            self.assertTrue(torch.allclose(y_pt, y, atol=1e-1, rtol=1e-1))

    def test_split_reshape_bmm_permute_rocm(self):
        self._test_split_reshape_bmm_permute(
            bs=[1], nheads=12, seq_len=256, hidden_size=768, test_name="static"
        )
        self._test_split_reshape_bmm_permute(
            bs=[16], nheads=12, seq_len=256, hidden_size=768, test_name="static"
        )


filter_test_cases_by_test_env(SplitBMMTestCase)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
