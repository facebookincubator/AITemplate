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
class GroupedClassicB2bBmmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def _test_grouped_classic_b2b_bmm(
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
        test_name="grouped_classic_b2b_bmm",
        alpha1_divide_by_seq_len=True,
        copy_op=True,
        atol=0.01,
        rtol=0.01,
        use_fp16_acc=False,
        random_seed=0,
    ):
        if isinstance(random_seed, list):
            random_seeds = random_seed
        else:
            random_seeds = [random_seed]
        # Initialize AIT classic_b2b_bmm operator.
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
        grouped_classic_b2b_bmm_op = ops.grouped_classic_b2b_bmm(
            causal_type=causal_type,
            alpha0=alpha0,
            alpha1=1.0,
            alpha1_divide_by_seq_len=alpha1_divide_by_seq_len,
            epilogue_math_name=epilogue_math_name,
        )
        if copy_op:
            grouped_classic_b2b_bmm_op = ops.grouped_classic_b2b_bmm(
                **grouped_classic_b2b_bmm_op._get_op_attributes()
            )
        Y = grouped_classic_b2b_bmm_op(Q, K, V, Bias)
        Y._attrs["is_output"] = True
        Y._attrs["name"] = "output"

        target = detect_target(use_fp16_acc=use_fp16_acc)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = use_fp16_acc
        module = compile_model(Y, target, "./tmp", test_name)
        # input(f"Connect debugger. {os.getpid()=}")
        # Run tests.
        torch_dtype = string_to_torch_dtype(dtype)
        offsets_torch_dtype = string_to_torch_dtype(offsets_dtype)
        y_results = {}
        for random_seed in random_seeds:
            torch.manual_seed(random_seed)
            for max_seq_len in sorted(set(max_seq_lens)):
                for num_head in sorted(set(num_heads)):
                    batch_sizes_sorted = sorted(set(batch_sizes), reverse=True)
                    max_batch_size = batch_sizes_sorted[0]
                    lengths_max = torch.randint(
                        1, max_seq_len, (max_batch_size + 1,), dtype=offsets_torch_dtype
                    )
                    lengths_max[0] = 0
                    offsets_max = torch.cumsum(lengths_max, dim=0).to(
                        dtype=offsets_torch_dtype
                    )
                    # print(f"{batch_size=}, {offsets=}")
                    total_length_max = offsets_max[-1]
                    offsets_max_pt = offsets_max.cuda()
                    q_pt_max = torch.rand(
                        (total_length_max, num_head, head_dim), dtype=torch_dtype
                    ).cuda()
                    k_pt_max = torch.rand(
                        (total_length_max, num_head, head_dim), dtype=torch_dtype
                    ).cuda()
                    v_pt_max = torch.rand(
                        (total_length_max, num_head, head_dim_value), dtype=torch_dtype
                    ).cuda()
                    ## TEMP DEBUG
                    for i in range(len(offsets_max) - 1):
                        start = offsets_max[i]
                        end = offsets_max[i + 1]
                        q_pt_max[start:end, :, :] += (
                            0.125 * i
                        )  # Making sure not everything averages out to zero
                        k_pt_max[start:end, :, :] += (
                            -0.25 * i + 0.15
                        )  # Making sure not everything averages out to zero
                        v_pt_max[start:end, :, :] += (
                            0.375 * i - 0.0125
                        )  # Making sure not everything averages out to zero
                    ## END TEMP DEBUG
                    bias_shape_max = [
                        max_batch_size,
                        num_head,
                        max_seq_len,
                        max_seq_len,
                    ]
                    if bias_broadcast:
                        for i, broadcast in enumerate(bias_broadcast):
                            if broadcast:
                                bias_shape_max[i] = 1
                    bias_pt_max = torch.rand(bias_shape_max, dtype=torch_dtype).cuda()
                    if not has_bias:
                        bias_pt_max *= 0.0
                    results_per_batch = {}
                    for batch_size in batch_sizes_sorted:
                        # Initialize inputs
                        # input(f"Attach debugger if you want. {os.getpid()=}. Press Enter to continue.")
                        total_length = offsets_max[batch_size]
                        q_pt = q_pt_max[:total_length, :, :].contiguous()
                        k_pt = k_pt_max[:total_length, :, :].contiguous()
                        v_pt = v_pt_max[:total_length, :, :].contiguous()
                        bias_pt = bias_pt_max[:batch_size, :, :, :].contiguous()
                        offsets_pt = offsets_max_pt[: batch_size + 1].contiguous()
                        # Run AIT.
                        inputs = {
                            "q": q_pt,
                            "k": k_pt,
                            "v": v_pt,
                            "offsets": offsets_pt,
                            "bias": bias_pt,
                        }
                        y = torch.empty(
                            [total_length, num_head, head_dim_value],
                            dtype=torch_dtype,
                            device="cuda",
                        )
                        ypadded = torch.zeros(
                            y.flatten().shape[0] + 128,
                            dtype=torch_dtype,
                            device=y.device,
                        )
                        y = ypadded[64:-64].reshape(
                            [total_length, num_head, head_dim_value]
                        )
                        module.run_with_tensors(inputs, [y])

                        y_results[(batch_size, max_seq_len, num_head)] = y
                        assert torch.all(
                            ypadded[:64] == 0
                        )  # Make sure we're not writing beyond boundaries
                        assert torch.all(
                            ypadded[-64:] == 0
                        )  # Make sure we're not writing beyond boundaries
                        # Run PT reference and verify results.
                        for row in range(batch_size):
                            start = offsets_max[row]
                            end = offsets_max[row + 1]
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
                                    0
                                    if (
                                        bias_broadcast is not None and bias_broadcast[0]
                                    )
                                    else row
                                )
                                bias_pt_row = bias_pt[
                                    bias_row : bias_row + 1, :, :length, :length
                                ].squeeze(dim=0)
                                attn = attn + bias_pt_row
                            attn = epilogue_math_name_to_torch_fn(epilogue_math_name)(
                                attn
                            )
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
                            results_per_batch[batch_size] = {
                                "y": y[start:end, :, :],
                                "expected_y": y_pt_row.to(torch_dtype),
                            }

                            torch.testing.assert_close(
                                y[start:end, :, :],
                                y_pt_row.to(torch_dtype),
                                atol=atol,
                                rtol=rtol,
                            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_1(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_1",
            dtype="float16",
            batch_sizes=1,
            head_dim=64,
            head_dim_value=64,
            max_seq_lens=[64],
            has_bias=False,
            random_seed=list(range(3)),
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_2(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_2",
            dtype="float16",
            batch_sizes=1,
            random_seed=list(range(3)),
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_3_batch_a(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_3_batch_a",
            dtype="float16",
            batch_sizes=4,
            random_seed=list(range(3)),
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_3_batch_b(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_3_batch_b",
            dtype="float16",
            batch_sizes=[2, 4],
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_3_batch_c(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_3_batch_c",
            dtype="float16",
            batch_sizes=2,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_3_batch_d(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_3_batch_d",
            dtype="float16",
            batch_sizes=[2, 33],
            random_seed=list(range(3)),
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_3_batch_e(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_3_batch_e",
            dtype="float16",
            batch_sizes=[2, 4],
            num_heads=[3, 5],
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_3_batch_f_bias(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="test_grouped_classic_b2b_bmm_fp16_3_batch_f_bias",
            dtype="float16",
            batch_sizes=[2, 4],
            num_heads=[3, 5],
            has_bias=True,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_3_batch_g_bias(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="test_grouped_classic_b2b_bmm_fp16_3_batch_g_bias",
            dtype="float16",
            batch_sizes=[2, 4],
            num_heads=[3, 5],
            has_bias=True,
            random_seed=list(range(3)),
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_acc(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="grouped_classic_b2b_bmm_fp16_acc",
            dtype="float16",
            batch_sizes=[7],
            use_fp16_acc=True,
            # Need to use a larger threshold for fp16 accum
            atol=0.25,
            random_seed=list(range(3)),
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty1(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty1",
            dtype="float16",
            batch_sizes=[5],
            num_heads=4,
            causal_type=CausalType.LOWER_LEFT_EMPTY,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty2(self):
        self._test_grouped_classic_b2b_bmm(
            test_name="test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty2",
            dtype="float16",
            batch_sizes=[1, 5, 33],
            num_heads=[
                2,
                4,
                11,
            ],
            causal_type=CausalType.LOWER_LEFT_EMPTY,
            random_seed=list(range(3)),
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty3_silu(self):
        for max_seq_len in [64, 256, 512]:
            self._test_grouped_classic_b2b_bmm(
                test_name=f"grouped_classic_b2b_bmm_fp16_causal_lower_left_empty_seqlen_{max_seq_len}",
                dtype="float16",
                batch_sizes=[1, 5, 33],
                max_seq_lens=max_seq_len,
                num_heads=[
                    2,
                    4,
                    11,
                ],
                epilogue_math_name="SiLu",
                causal_type=CausalType.LOWER_LEFT_EMPTY,
                random_seed=list(range(10)),
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty3_simple_silu_bias(
        self,
    ):
        for max_seq_len in [512]:
            self._test_grouped_classic_b2b_bmm(
                test_name=f"grouped_classic_b2b_bmm_fp16_causal_lower_left_empty_seqlen_{max_seq_len}_silu_bias",
                dtype="float16",
                batch_sizes=[3, 33],
                max_seq_lens=max_seq_len,
                num_heads=[
                    11,
                ],
                epilogue_math_name="SiLu",
                causal_type=CausalType.LOWER_LEFT_EMPTY,
                random_seed=list(range(3)),
                has_bias=True,
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty3_simple_silu_bias_broadcast_1(
        self,
    ):
        for max_seq_len in [512]:
            for random_seed in range(1):
                self._test_grouped_classic_b2b_bmm(
                    test_name=f"grouped_classic_b2b_bmm_fp16_causal_lower_left_empty_seqlen_{max_seq_len}_seed{random_seed}_bias_broadcast_1",
                    dtype="float16",
                    batch_sizes=[3, 33],
                    max_seq_lens=max_seq_len,
                    num_heads=[
                        11,
                    ],
                    epilogue_math_name="SiLu",
                    causal_type=CausalType.LOWER_LEFT_EMPTY,
                    random_seed=random_seed,
                    has_bias=True,
                    bias_broadcast=[True, False, True, False],
                )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty3_simple_silu_bias_broadcast_2(
        self,
    ):
        for max_seq_len in [512]:
            for random_seed in range(1):
                self._test_grouped_classic_b2b_bmm(
                    test_name=f"grouped_classic_b2b_bmm_fp16_causal_lower_left_empty_seqlen_{max_seq_len}_seed{random_seed}_bias_broadcast_2",
                    dtype="float16",
                    batch_sizes=[3, 33],
                    max_seq_lens=max_seq_len,
                    num_heads=[
                        11,
                    ],
                    epilogue_math_name="SiLu",
                    causal_type=CausalType.LOWER_LEFT_EMPTY,
                    random_seed=random_seed,
                    has_bias=True,
                    bias_broadcast=[True, False, False, False],
                )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_causal_lower_left_empty3_simple_silu_bias_broadcast_3(
        self,
    ):
        for max_seq_len in [512]:
            self._test_grouped_classic_b2b_bmm(
                test_name=f"grouped_classic_b2b_bmm_fp16_causal_lower_left_empty_seqlen_{max_seq_len}_bias_broadcast_3",
                dtype="float16",
                batch_sizes=[3, 33],
                max_seq_lens=max_seq_len,
                num_heads=[
                    11,
                ],
                epilogue_math_name="SiLu",
                causal_type=CausalType.LOWER_LEFT_EMPTY,
                random_seed=list(range(12, 24)),
                has_bias=True,
                bias_broadcast=[True, True, True, False],
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_large_bias_broadcast_4(
        self,
    ):
        for max_seq_len in [64, 256, 512]:
            self._test_grouped_classic_b2b_bmm(
                test_name=f"test_grouped_classic_b2b_bmm_fp16_large_bias_broadcast_4_seqlen={max_seq_len}",
                dtype="float16",
                batch_sizes=[3, 33],
                max_seq_lens=max_seq_len,
                num_heads=[
                    2,
                    11,
                ],
                random_seed=list(range(5)),
                has_bias=True,
                bias_broadcast=[True, False, True, False],
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_grouped_classic_b2b_bmm_fp16_large_bias_broadcast_5(
        self,
    ):
        self._test_grouped_classic_b2b_bmm(
            test_name="test_grouped_classic_b2b_bmm_fp16_large_bias_broadcast_5",
            dtype="float16",
            batch_sizes=[3, 33],
            max_seq_lens=256,
            num_heads=[
                2,
                11,
            ],
            random_seed=list(range(3400, 3411)),
            has_bias=True,
            bias_broadcast=[True, False, True, False],
        )


if __name__ == "__main__":
    unittest.main()
