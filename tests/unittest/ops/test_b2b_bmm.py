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
import itertools
import logging
import unittest
from typing import List, Tuple

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import CausalType
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    epilogue_math_name_to_torch_fn,
    get_attn_mask_per_causal_type,
)
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
        causal_type=CausalType.NO_CAUSAL,
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
        alpha1 = 1.0
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
            causal_type=causal_type,
            alpha0=alpha0,
            alpha1=alpha1,
            alpha1_divide_by_seq_len=True,
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
            attn = epilogue_math_name_to_torch_fn(epilogue_math_name)(attn)
            attn = alpha1 / m * attn
            invalid_attn_mask = get_attn_mask_per_causal_type(
                m, n0, causal_type, torch_dtype
            )
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
            batch_sizes=1,
        )
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_dynamic_batch",
            dtype="float16",
            batch_sizes=[3, 8, 10],
        )
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_rectangular",
            dtype="float16",
            batch_sizes=[2],
            m=512,
            n0=128,
            n1=128,
        )
        self._test_classic_b2b_bmm(
            test_name="classic_b2b_bmm_fp16_causal",
            dtype="float16",
            batch_sizes=5,
            causal_type=CausalType.LOWER_LEFT_EMPTY,
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
            epilogue_math_name="ReLu",
            causal_type=CausalType.LOWER_LEFT_EMPTY,
        )


@unittest.skipIf(
    detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
    "Not supported by CUDA < SM80.",
)
class FMHAStyleB2bBmmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def _test_fmha_style_b2b_bmm(
        self,
        batch_sizes: Tuple[int, List[int]] = 1024,
        seq_lens: Tuple[int, List[int]] = 256,
        k0=128,
        seq_lens_kv: Tuple[int, List[int]] = 256,
        n1=256,
        num_heads: Tuple[int, List[int]] = 1,
        has_bias=False,
        bias_broadcast=None,
        epilogue_math_name="Identity",
        causal_type=CausalType.NO_CAUSAL,
        dtype="float16",
        test_name="fmha_style_b2b_bmm",
        copy_op=True,
        atol=1e-3,
        rtol=1e-2,
        use_fp16_acc=True,
    ):
        # Initialize AIT fmha_style_b2b_bmm operator.
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]
        if isinstance(seq_lens, int):
            seq_lens = [seq_lens]
        if isinstance(seq_lens_kv, int):
            seq_lens_kv = [seq_lens_kv]
        if isinstance(num_heads, int):
            num_heads = [num_heads]
        alpha0 = 1.0 / (k0**0.5)
        alpha1 = 1.0
        batch_size_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_size")
        seq_lens_dim = shape_utils.gen_int_var_min_max(seq_lens, "seq_len")
        seq_lens_kv_dim = shape_utils.gen_int_var_min_max(seq_lens_kv, "seq_len_kv")
        num_heads_dim = shape_utils.gen_int_var_min_max(num_heads, "num_heads")

        Q = Tensor(
            shape=[batch_size_dim, seq_lens_dim, num_heads_dim, k0],
            dtype=dtype,
            name="q",
            is_input=True,
        )
        K = Tensor(
            shape=[batch_size_dim, seq_lens_kv_dim, num_heads_dim, k0],
            dtype=dtype,
            name="k",
            is_input=True,
        )
        V = Tensor(
            shape=[batch_size_dim, seq_lens_kv_dim, num_heads_dim, n1],
            dtype=dtype,
            name="v",
            is_input=True,
        )
        Bias = None
        if has_bias:
            shape = [batch_size_dim, num_heads_dim, seq_lens_dim, seq_lens_kv_dim]
            if bias_broadcast:
                for i, broadcast in enumerate(bias_broadcast):
                    if broadcast:
                        shape[i] = 1
            Bias = Tensor(
                shape=shape,
                dtype=dtype,
                name="bias",
                is_input=True,
            )
        fmha_style_b2b_bmm_op = ops.fmha_style_b2b_bmm(
            causal_type=causal_type,
            alpha0=alpha0,
            alpha1=alpha1,
            alpha1_divide_by_seq_len=True,
            epilogue_math_name=epilogue_math_name,
        )
        if copy_op:
            fmha_style_b2b_bmm_op = ops.fmha_style_b2b_bmm(
                **fmha_style_b2b_bmm_op._get_op_attributes()
            )
        Y = fmha_style_b2b_bmm_op(Q, K, V, Bias)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        target = detect_target(use_fp16_acc=use_fp16_acc)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = use_fp16_acc
        module = compile_model(Y, target, "./tmp", test_name)

        # Run tests.
        torch_dtype = string_to_torch_dtype(dtype)
        for batch_size, seq_len, seq_len_kv, num_head in itertools.product(
            batch_sizes, seq_lens, seq_lens_kv, num_heads
        ):
            # Initialize inputs
            q_pt = torch.rand(
                batch_size, seq_len, num_head, k0, dtype=torch_dtype
            ).cuda()
            k_pt = torch.rand(
                batch_size, seq_len_kv, num_head, k0, dtype=torch_dtype
            ).cuda()
            v_pt = torch.rand(
                batch_size, seq_len_kv, num_head, n1, dtype=torch_dtype
            ).cuda()
            shape = [batch_size, num_head, seq_len, seq_len_kv]
            if bias_broadcast:
                for i, broadcast in enumerate(bias_broadcast):
                    if broadcast:
                        shape[i] = 1
            bias_pt = torch.rand(shape, dtype=torch_dtype).cuda()

            # Run PT reference.
            attn = alpha0 * (
                q_pt.transpose(1, 2) @ k_pt.transpose(1, 2).transpose(-2, -1)
            )
            if has_bias:
                attn = attn + bias_pt
            attn = epilogue_math_name_to_torch_fn(epilogue_math_name)(attn)
            attn = alpha1 / seq_len * attn
            invalid_attn_mask = get_attn_mask_per_causal_type(
                seq_len, seq_len_kv, causal_type, torch_dtype
            )
            attn = attn * invalid_attn_mask
            output = (attn @ v_pt.transpose(1, 2)).transpose(1, 2)
            y_pt = output.detach()

            # Run AIT.
            inputs = {
                "q": q_pt,
                "k": k_pt,
                "v": v_pt,
            }
            if has_bias:
                inputs["bias"] = bias_pt
            y = torch.empty(
                [batch_size, seq_len, num_head, n1],
                dtype=torch_dtype,
                device="cuda",
            )
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(y, y_pt.to(torch_dtype), atol=atol, rtol=rtol)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_fmha_style_b2b_bmm_fp16(self):
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_basic",
            dtype="float16",
            batch_sizes=1,
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_dynamic_batch",
            dtype="float16",
            batch_sizes=[3, 8, 10],
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_dynamic_seq_len",
            dtype="float16",
            seq_lens=[128, 256],
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_dynamic_seq_len_kv",
            dtype="float16",
            seq_lens_kv=[128, 256],
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_dynamic_num_heads",
            dtype="float16",
            num_heads=[1, 2],
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_rectangular",
            dtype="float16",
            batch_sizes=[2],
            seq_lens=512,
            seq_lens_kv=128,
            n1=128,
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_causal_upper_right_empty",
            dtype="float16",
            batch_sizes=2,
            causal_type=CausalType.UPPER_RIGHT_EMPTY,
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_causal_lower_left_empty",
            dtype="float16",
            batch_sizes=3,
            causal_type=CausalType.LOWER_LEFT_EMPTY,
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_bias",
            dtype="float16",
            batch_sizes=2,
            has_bias=True,
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_bias_broadcast",
            dtype="float16",
            batch_sizes=3,
            has_bias=True,
            bias_broadcast=[False, True, False, False],
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_sigmoid",
            dtype="float16",
            batch_sizes=[1, 4],
            epilogue_math_name="Sigmoid",
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_multi_head",
            dtype="float16",
            batch_sizes=[1, 4],
            has_bias=True,
            num_heads=2,
            bias_broadcast=[True, True, True, False],
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_complex",
            dtype="float16",
            batch_sizes=[1, 4],
            epilogue_math_name="SiLu",
            causal_type=CausalType.LOWER_LEFT_EMPTY,
            has_bias=True,
            bias_broadcast=[False, True, False, False],
            num_heads=4,
        )
        self._test_fmha_style_b2b_bmm(
            test_name="fmha_style_b2b_bmm_fp16_complex_fp32_acc",
            dtype="float16",
            batch_sizes=[1, 4],
            epilogue_math_name="ReLu",
            causal_type=CausalType.LOWER_LEFT_EMPTY,
            has_bias=True,
            bias_broadcast=[False, False, True, False],
            num_heads=2,
            use_fp16_acc=False,
            seq_lens=512,
            seq_lens_kv=512,
        )


if __name__ == "__main__":
    unittest.main()
