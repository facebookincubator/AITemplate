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
import itertools
import unittest
from typing import Tuple

import torch

from aitemplate.compiler import compile_model, ops

from aitemplate.compiler.base import IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils import shape_utils

from parameterized import parameterized


class FuseBmmPermuteCase(unittest.TestCase):
    def _create_bmm_permute_graph(
        self,
        A_shape: Tuple[IntVar, int, int],
        B_shape: Tuple[IntVar, int, int],
        bmm_type: str,
        dtype: str,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Create a graph consisting of bmm with given layout + permute021.
        """
        OP = getattr(ops, bmm_type, None)
        assert OP is not None

        A = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        B = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        inputs = [A, B]

        Y = OP()(*inputs)
        Y = ops.permute021()(Y)
        Y._attrs["name"] = "target_bmm_tensor"
        return A, B, Y

    def _test_bmm_permute(
        self,
        B: int,
        A_shape: Tuple[IntVar, int, int],
        B_shape: Tuple[IntVar, int, int],
        orig_layout: str,
        dtype: str = "float16",
    ):

        is_row_major_a = orig_layout[0] == "r"
        is_row_major_b = orig_layout[1] == "r"
        is_row_major_c = orig_layout[2] == "r"

        new_layout = orig_layout[:2] + ("c" if is_row_major_c else "r")
        testname = f"{orig_layout}_to_{new_layout}_{dtype}"

        original_bmm = f"bmm_{orig_layout}"
        new_bmm = f"bmm_{new_layout}"

        X, W, bmm_tensor = self._create_bmm_permute_graph(
            A_shape,
            B_shape,
            original_bmm,
            dtype,
        )

        output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        # Check that the new bmm is present and the original is not
        exist_new_bmm = False
        for tensor in module.debug_sorted_graph:
            src_ops = tensor.src_ops()
            if len(src_ops) == 0:
                continue
            assert (
                len(src_ops) == 1
            ), "Constructed graph should only have single-source op tensors."
            src_op = list(tensor.src_ops())[0]
            assert src_op._attrs["op"] != original_bmm

            if src_op._attrs["op"] == new_bmm:
                exist_new_bmm = True

        assert exist_new_bmm, "Can't find converted bmm op in the graph."

        m = A_shape[-2] if is_row_major_a else A_shape[-1]
        n = B_shape[-1] if is_row_major_b else B_shape[-2]
        k = B_shape[-2] if is_row_major_b else B_shape[-1]

        # Check that fused graph produces correct output
        for b in B:
            # Compute PyTorch output
            X_pt = get_random_torch_tensor((b, m, k), dtype)
            W_pt = get_random_torch_tensor((b, k, n), dtype)
            Y_pt = torch.matmul(X_pt, W_pt)
            if is_row_major_c:
                Y_pt = torch.transpose(Y_pt, 2, 1)
            Y_pt = torch.cos(Y_pt)

            # Compute AIT output
            out_shape = [b, m, n] if not is_row_major_c else [b, n, m]
            y = get_torch_empty_tensor(out_shape, dtype)
            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0]
            if not is_row_major_a:
                X_pt = torch.transpose(X_pt, 2, 1).contiguous()
            if not is_row_major_b:
                W_pt = torch.transpose(W_pt, 2, 1).contiguous()
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            module.run_with_tensors(inputs, [y])

            torch.testing.assert_close(Y_pt, y, atol=1e-1, rtol=1e-1)

    @parameterized.expand(
        itertools.product(
            [[1, 4]],  # Batch size
            ["r", "c"],  # Layout of A
            ["r", "c"],  # Layout of B
            ["r", "c"],  # Layout of output
            filter_test_cases_by_params(
                {
                    TestEnv.CUDA_LESS_THAN_SM80: ["float16"],
                }
            ),
        )
    )
    def test_xxr_to_xxс(self, B, layout_a, layout_b, layout_c, dtype):
        """
        Test that bmm_xxr + permute021 is fused into bmm_xxc and the other way round.
        """
        M, N, K = 4, 6, 8
        batch_dim = shape_utils.gen_int_var_min_max(B)

        shape_a = [batch_dim, K, M] if layout_a == "c" else [batch_dim, M, K]
        shape_b = [batch_dim, N, K] if layout_b == "c" else [batch_dim, K, N]

        self._test_bmm_permute(
            B,
            shape_a,
            shape_b,
            layout_a + layout_b + layout_c,
            dtype=dtype,
        )
