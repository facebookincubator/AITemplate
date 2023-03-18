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
Unittests for batched_dense_vec_jagged_2d_mul Operator.
"""
import unittest
from typing import List

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar, JaggedDim, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.jagged_utils import batched_dense_vec_jagged_2d_mul_ref
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import parameterized


_TOLERANCE_LIMITS = {
    "float16": {"atol": 1e-1, "rtol": 1e-1},
    "float32": {"atol": 3e-2, "rtol": 2e-2},
    "bfloat16": {"atol": 2e-1, "rtol": 2e-1},
}


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class BatchedDenseVecJagged2DMulTestCase(unittest.TestCase):
    def _test_batched_dense_vec_jagged_2d_mul(
        self,
        B: int,
        N: int,
        H: int,
        D: int,
        offsets: List[int],
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        use_fp16_acc: bool = False,
        test_suffix: str = "",
    ):
        # jagged shape is equal to (B, N, H, D)
        batch_size = B
        batch_dim = IntVar(values=[1, batch_size * 2], name="batch_size")
        jagged_dims = [JaggedDim(min_value=0, max_value=N)]

        total_length = offsets[-1]
        total_length_dim = IntVar(values=[1, total_length * 2], name="total_length")
        jagged_inner_shape = [H, D]
        jagged_inner_dims = [IntImm(dim) for dim in jagged_inner_shape]
        jagged_input_shape = [total_length] + jagged_inner_shape

        offsets_dim = IntVar(values=[2, len(offsets) * 2])

        # dense shape is (B, H, N)
        dense_shape = [batch_size, H, N]
        dense_dims = [batch_dim, IntImm(H), IntImm(N)]
        SOURCE = Tensor(
            shape=[
                total_length_dim,
                *jagged_inner_dims,
            ],
            name="source",
            dtype=dtype,
            is_input=True,
        )
        OFFSETS_LIST = [
            Tensor(
                shape=[
                    offsets_dim,
                ],
                name="offsets",
                dtype=offsets_dtype,
                is_input=True,
            )
        ]
        JAGGED = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=jagged_dims,
        )(SOURCE, OFFSETS_LIST)

        DENSE = Tensor(
            shape=dense_dims,
            name="dense",
            dtype=dtype,
            is_input=True,
        )

        RESULT = ops.batched_dense_vec_jagged_2d_mul()(DENSE, JAGGED)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        assert not SOURCE.is_jagged()
        assert not DENSE.is_jagged()
        assert JAGGED.is_jagged()
        assert not RESULT.is_jagged()

        model = compile_model(
            [RESULT],
            detect_target(use_fp16_acc=use_fp16_acc),
            "./tmp",
            f"test_batched_dense_vec_jagged_2d_mul_{test_suffix}",
        )

        torch_offsets_dtype = string_to_torch_dtype(offsets_dtype)
        offsets_pt = torch.tensor(offsets, dtype=torch_offsets_dtype).cuda()
        source_pt = get_random_torch_tensor(jagged_input_shape, dtype)
        dense_pt = get_random_torch_tensor(dense_shape, dtype)
        result_pt = batched_dense_vec_jagged_2d_mul_ref(
            vectors=dense_pt,
            matrices=source_pt,
            offsets=offsets_pt,
        )
        result = get_torch_empty_tensor([batch_size, H, D], dtype)

        inputs = {"dense": dense_pt, "source": source_pt, "offsets": offsets_pt}
        model.run_with_tensors(inputs, [result])

        tolerance_limits = _TOLERANCE_LIMITS[dtype]
        torch.testing.assert_close(result, result_pt, **tolerance_limits)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                # TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_batched_dense_vesc_jagged_2d_mul(self, ait_dtype):
        # test with different combination of offsets_dtype, use_fp16_acc and shapes
        self._test_batched_dense_vec_jagged_2d_mul(
            4,
            260,
            10,
            32,
            [0, 1, 4, 6, 7],
            dtype=ait_dtype,
            offsets_dtype="int32",
            use_fp16_acc=True,
            test_suffix=f"{ait_dtype}_int32_True",
        )
        self._test_batched_dense_vec_jagged_2d_mul(
            6,
            130,
            15,
            39,
            [0, 1, 4, 6, 7, 9, 10],
            dtype=ait_dtype,
            offsets_dtype="int32",
            use_fp16_acc=False,
            test_suffix=f"{ait_dtype}_int32_False",
        )
        self._test_batched_dense_vec_jagged_2d_mul(
            8,
            52,
            21,
            32,
            [0, 1, 4, 6, 7, 8, 12, 20, 29],
            dtype=ait_dtype,
            offsets_dtype="int64",
            use_fp16_acc=True,
            test_suffix=f"{ait_dtype}_int64_True",
        )
        self._test_batched_dense_vec_jagged_2d_mul(
            10,
            10,
            32,
            8,
            [0, 1, 4, 6, 7, 11, 15, 19, 23, 26, 28],
            dtype=ait_dtype,
            offsets_dtype="int64",
            use_fp16_acc=False,
            test_suffix=f"{ait_dtype}_int64_False",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
