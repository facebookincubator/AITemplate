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
Unittests for grouped b2b bmm Operators.
"""
import itertools
import logging
import unittest
from typing import List, Tuple

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntVar, JaggedDim
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
class GroupedFMHAStyleB2bBmmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def _test_grouped_fmha_style_b2b_bmm(
        self,
        batch_sizes: Tuple[int, List[int]] = 1024,
        max_seq_lens: Tuple[int, List[int]] = 256,
        head_dim=128,
        head_dim_value=256,
        num_heads: Tuple[int, List[int]] = 1,
        has_bias=False,
        bias_broadcast=None,
        epilogue_math_name="Identity",
        causal_type=CausalType.NO_CAUSAL,
        dtype="float16",
        offsets_dtype="int32",
        test_name="grouped_fmha_style_b2b_bmm",
        alpha1_divide_by_seq_len=True,
        copy_op=True,
        atol=1e-3,
        rtol=1e-3,
        use_fp16_acc=False,
    ):
        # Initialize AIT fmha_style_b2b_bmm operator.
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes, batch_sizes]
        if isinstance(max_seq_lens, int):
            max_seq_lens = [max_seq_lens, max_seq_lens]
        if isinstance(num_heads, int):
            num_heads = [num_heads, num_heads]
        alpha0 = 1.0 / (head_dim**0.5)
        batch_size_dim = IntVar(
            values=[min(batch_sizes), max(batch_sizes)], name="batch_size"
        )
        max_seq_len_dim = shape_utils.gen_int_var_min_max(
            max_seq_lens, name="max_seq_len"
        )
        num_heads_dim = shape_utils.gen_int_var_min_max(num_heads, name="num_heads")
        jagged_dims = [JaggedDim(min_value=0, max_value=max_seq_len_dim)]
        total_length_dim = IntVar(
            values=[0, batch_size_dim.upper_bound() * max_seq_len_dim.upper_bound()],
            name="total_length",
        )
        offsets_dim = IntVar(
            values=[batch_size_dim.lower_bound() + 1, batch_size_dim.upper_bound() + 1],
            name="offset_length",
        )
        Q_dense = Tensor(
            shape=[total_length_dim, num_heads_dim, head_dim],
            dtype=dtype,
            name="q",
            is_input=True,
        )
        K_dense = Tensor(
            shape=[total_length_dim, num_heads_dim, head_dim],
            dtype=dtype,
            name="k",
            is_input=True,
        )
        V_dense = Tensor(
            shape=[total_length_dim, num_heads_dim, head_dim_value],
            dtype=dtype,
            name="v",
            is_input=True,
        )
        offsets = [
            Tensor(
                shape=[offsets_dim], name="offsets", dtype=offsets_dtype, is_input=True
            )
        ]
        Q = ops.make_jagged(batch_dim=batch_size_dim, jagged_dims=jagged_dims)(
            Q_dense, offsets
        )
        K = ops.make_jagged(batch_dim=batch_size_dim, jagged_dims=jagged_dims)(
            K_dense, offsets
        )
        V = ops.make_jagged(batch_dim=batch_size_dim, jagged_dims=jagged_dims)(
            V_dense, offsets
        )
        Bias = None
        if has_bias:
            shape = [batch_size_dim, num_heads_dim, max_seq_len_dim, max_seq_len_dim]
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
        grouped_fmha_style_b2b_bmm_op = ops.grouped_fmha_style_b2b_bmm(
            causal_type=causal_type,
            alpha0=alpha0,
            alpha1=1.0,
            alpha1_divide_by_seq_len=alpha1_divide_by_seq_len,
            epilogue_math_name=epilogue_math_name,
        )
        if copy_op:
            grouped_fmha_style_b2b_bmm_op = ops.grouped_fmha_style_b2b_bmm(
                **grouped_fmha_style_b2b_bmm_op._get_op_attributes()
            )
        Y = grouped_fmha_style_b2b_bmm_op(Q, K, V, Bias)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        target = detect_target(use_fp16_acc=use_fp16_acc)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = use_fp16_acc
        module = compile_model(Y, target, "./tmp", test_name)

        # Run tests.
        torch_dtype = string_to_torch_dtype(dtype)
        offsets_torch_dtype = string_to_torch_dtype(offsets_dtype)
        for batch_size, max_seq_len, num_head in itertools.product(
            sorted(set(batch_sizes)), sorted(set(max_seq_lens)), sorted(set(num_heads))
        ):
            # Initialize inputs
            lengths = torch.randint(
                1, max_seq_len, (batch_size + 1,), dtype=offsets_torch_dtype
            )
            lengths[0] = 0
            offsets = torch.cumsum(lengths, dim=0).to(dtype=offsets_torch_dtype)
            # print(f"{batch_size=}, {offsets=}")
            total_length = offsets[-1]
            offsets_pt = offsets.cuda()
            q_pt = torch.rand(
                (total_length, num_head, head_dim), dtype=torch_dtype
            ).cuda()
            k_pt = torch.rand(
                (total_length, num_head, head_dim), dtype=torch_dtype
            ).cuda()
            v_pt = torch.rand(
                (total_length, num_head, head_dim_value), dtype=torch_dtype
            ).cuda()
            bias_shape = [batch_size, num_head, max_seq_len, max_seq_len]
            if bias_broadcast:
                for i, broadcast in enumerate(bias_broadcast):
                    if broadcast:
                        bias_shape[i] = 1
            bias_pt = torch.rand(bias_shape, dtype=torch_dtype).cuda()

            # Run AIT.
            inputs = {
                "q": q_pt,
                "k": k_pt,
                "v": v_pt,
                "offsets": offsets_pt,
            }
            if has_bias:
                inputs["bias"] = bias_pt
            y = torch.empty(
                [total_length, num_head, head_dim_value],
                dtype=torch_dtype,
                device="cuda",
            )
            module.run_with_tensors(inputs, [y])

            # Run PT reference and verify results.
            for row in range(batch_size):
                start = offsets[row]
                end = offsets[row + 1]
                length = end - start
                q_pt_row = q_pt[start:end, :, :]
                k_pt_row = k_pt[start:end, :, :]
                v_pt_row = v_pt[start:end, :, :]
                attn = alpha0 * (
                    q_pt_row.transpose(0, 1)
                    @ k_pt_row.transpose(0, 1).transpose(-2, -1)
                )
                if has_bias:
                    bias_row = (
                        0 if (bias_broadcast is not None and bias_broadcast[0]) else row
                    )
                    bias_pt_row = bias_pt[
                        bias_row : bias_row + 1, :, :length, :length
                    ].squeeze(dim=0)
                    attn = attn + bias_pt_row
                attn = epilogue_math_name_to_torch_fn(epilogue_math_name)(attn)
                if alpha1_divide_by_seq_len:
                    attn /= max_seq_len
                invalid_attn_mask = get_attn_mask_per_causal_type(
                    length, length, causal_type, torch_dtype
                )
                attn = attn * invalid_attn_mask
                output = (attn @ v_pt_row.transpose(0, 1)).transpose(0, 1)
                y_pt_row = output.detach()
                # print(
                #     f"{batch_size=}, {row=}, {y[start:end, :, :]=}, {y_pt_row.to(torch_dtype)=}"
                # )
                torch.testing.assert_close(
                    y[start:end, :, :], y_pt_row.to(torch_dtype), atol=atol, rtol=rtol
                )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_fmha_style_b2b_bmm_fp16(self):
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_basic",
            dtype="float16",
            batch_sizes=1,
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_dynamic_batch",
            dtype="float16",
            batch_sizes=[3, 8, 10],
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_dynamic_batch_fp16_acc",
            dtype="float16",
            batch_sizes=[3, 8, 10],
            use_fp16_acc=True,
            # Need to use a larger threshold for fp16 accum, it seems that
            # torch always generates the same result regardless of
            # how allow_fp16_reduced_precision_reduction is set.
            atol=1e-2,
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_causal_upper_right_empty",
            dtype="float16",
            batch_sizes=2,
            causal_type=CausalType.UPPER_RIGHT_EMPTY,
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_causal_lower_left_empty",
            dtype="float16",
            batch_sizes=3,
            causal_type=CausalType.LOWER_LEFT_EMPTY,
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_bias",
            dtype="float16",
            batch_sizes=2,
            has_bias=True,
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_bias_broadcast",
            dtype="float16",
            batch_sizes=3,
            has_bias=True,
            bias_broadcast=[False, True, False, False],
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_dynamic_seq_len",
            dtype="float16",
            max_seq_lens=[128, 256],
            has_bias=True,
            bias_broadcast=[False, True, False, False],
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_sigmoid",
            dtype="float16",
            batch_sizes=[1, 4],
            epilogue_math_name="Sigmoid",
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_multi_head",
            dtype="float16",
            batch_sizes=[1, 4],
            has_bias=True,
            num_heads=2,
            bias_broadcast=[True, True, True, False],
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_dynamic_multi_head",
            dtype="float16",
            num_heads=[2, 4],
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_complex",
            dtype="float16",
            offsets_dtype="int64",
            batch_sizes=[3, 4],
            epilogue_math_name="SiLu",
            causal_type=CausalType.LOWER_LEFT_EMPTY,
            has_bias=True,
            bias_broadcast=[False, True, False, False],
            num_heads=4,
        )
        self._test_grouped_fmha_style_b2b_bmm(
            test_name="grouped_fmha_style_b2b_bmm_fp16_complex_fp16_acc",
            dtype="float16",
            batch_sizes=[1, 4, 10, 512, 1024],
            epilogue_math_name="ReLu",
            causal_type=CausalType.LOWER_LEFT_EMPTY,
            has_bias=True,
            bias_broadcast=[False, False, True, False],
            num_heads=2,
            use_fp16_acc=True,
            max_seq_lens=1024,
            # Need to use a larger threshold for fp16 accum, it seems that
            # torch always generates the same result regardless of
            # how allow_fp16_reduced_precision_reduction is set.
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
