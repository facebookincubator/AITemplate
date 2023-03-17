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
"""
Unittests for b2b bmm Operators.
"""
import logging
import unittest
from typing import List, Tuple

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils
from aitemplate.utils.torch_utils import string_to_torch_dtype


_LOGGER = logging.getLogger(__name__)


@unittest.skipIf(
    detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
    "Not supported by CUDA < SM80.",
)
class ClassicB2bBmmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def _test_classic_b2b_bmm(
        self,
        batch_sizes: Tuple[int, List[int]] = 1024,
        m=256,
        k0=128,
        n0=256,
        n1=256,
        epilogue_math_name="Identity",
        causal=False,
        dtype="float16",
        test_name="classic_b2b_bmm",
        copy_op=True,
        atol=1e-2,
        rtol=1e-2,
    ):
        # Initialize AIT classic_b2b_bmm operator.
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]
        alpha0 = 1.0 / (k0**0.5)
        alpha1 = 1.0 / m
        batch_size_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_size")

        Q = Tensor(
            shape=[batch_size_dim, m, k0],
            dtype=dtype,
            name="q",
            is_input=True,
        )
        K = Tensor(
            shape=[batch_size_dim, n0, k0],
            dtype=dtype,
            name="k",
            is_input=True,
        )
        V = Tensor(
            shape=[batch_size_dim, n0, n1],
            dtype=dtype,
            name="v",
            is_input=True,
        )
        Bias = Tensor(
            shape=[batch_size_dim, m, n0],
            dtype=dtype,
            name="bias",
            is_input=True,
        )
        classic_b2b_bmm_op = ops.classic_b2b_bmm(
            causal=causal,
            alpha0=alpha0,
            alpha1=alpha1,
            epilogue_math_name=epilogue_math_name,
        )
        if copy_op:
            classic_b2b_bmm_op = ops.classic_b2b_bmm(
                **classic_b2b_bmm_op._get_op_attributes()
            )
        Y = classic_b2b_bmm_op(Q, K, V, Bias)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        target = detect_target(use_fp16_acc=True)
        module = compile_model(Y, target, "./tmp", test_name)

        # Run tests.
        torch_dtype = string_to_torch_dtype(dtype)
        for batch_size in batch_sizes:
            # Initialize inputs
            q_pt = torch.rand(batch_size, m, k0, dtype=torch_dtype).cuda()
            k_pt = torch.rand(batch_size, n0, k0, dtype=torch_dtype).cuda()
            v_pt = torch.rand(batch_size, n0, n1, dtype=torch_dtype).cuda()
            bias_pt = torch.rand(batch_size, m, n0, dtype=torch_dtype).cuda()

            # Run PT reference.
            attn = alpha0 * (q_pt @ k_pt.transpose(-2, -1)) + bias_pt
            if epilogue_math_name == "Identity":
                pass
            elif epilogue_math_name == "Sigmoid":
                attn = torch.sigmoid(attn)
            elif epilogue_math_name == "SiLu":
                attn = torch.nn.functional.silu(attn)
            else:
                raise NotImplementedError(f"Unsupported {epilogue_math_name=}!")
            attn = alpha1 * attn
            if causal:
                invalid_attn_mask: torch.Tensor = 1.0 - torch.tril(
                    torch.ones(
                        (m, n0),
                        dtype=torch.bool,
                        device="cuda",
                    )
                ).fill_diagonal_(False).to(torch_dtype)
                attn = attn * invalid_attn_mask
            output = attn @ v_pt
            y_pt = output.detach()

            # Run AIT.
            inputs = {"q": q_pt, "k": k_pt, "v": v_pt, "bias": bias_pt}
            y = torch.empty(
                [batch_size, m, n1],
                dtype=torch_dtype,
                device="cuda",
            )
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y, y_pt.to(torch_dtype), atol=atol, rtol=rtol)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_classic_b2b_bmm_fp16(self):
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_basic",
            dtype="float16",
        )
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_dynamic_batch",
            dtype="float16",
            batch_sizes=[3, 8, 10],
        )
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_causal",
            dtype="float16",
            batch_sizes=5,
            causal=True,
        )
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_sigmoid",
            dtype="float16",
            batch_sizes=[1, 4],
            epilogue_math_name="Sigmoid",
        )
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_complex",
            dtype="float16",
            batch_sizes=[1, 4],
            epilogue_math_name="SiLu",
            causal=True,
        )


if __name__ == "__main__":
    unittest.main()
