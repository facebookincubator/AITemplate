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
Unittests for the jagged_to_padded_dense op.
"""

import json
import random
import tempfile
import unittest
from typing import List

import aitemplate.testing.jagged_utils as jagged_utils

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import JaggedDim
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import param, parameterized


class JaggedToPaddedDenseTestCase(unittest.TestCase):
    def _test_jagged_to_padded_dense(
        self,
        jagged_max_shape: List[int],
        offsets_list: List[List[int]],
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        test_suffix: str = "",
        padding_value: float = 0.0,
    ):
        batch_size = jagged_max_shape[0]
        batch_dim = IntVar(values=[1, batch_size * 2], name="batch_size")
        jagged_dims_max_values = jagged_max_shape[1 : 1 + len(offsets_list)]
        jagged_dims = [
            JaggedDim(min_value=0, max_value=max_value)
            for max_value in jagged_dims_max_values
        ]

        total_length = offsets_list[-1][-1]
        total_length_dim = IntVar(values=[1, total_length * 2], name="total_length")

        jagged_inner_shape = jagged_max_shape[1 + len(offsets_list) :]
        jagged_inner_dims = [IntImm(dim) for dim in jagged_inner_shape]
        jagged_input_shape = [total_length] + jagged_inner_shape

        offsets_dims = [
            IntVar(values=[2, len(offsets) * 2]) for offsets in offsets_list
        ]

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
                name=f"offsets{i}",
                dtype=offsets_dtype,
                is_input=True,
            )
            for i, offsets_dim in enumerate(offsets_dims)
        ]
        JAGGED = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=jagged_dims,
        )(SOURCE, OFFSETS_LIST)

        RESULT = ops.jagged_to_padded_dense(padding_value=padding_value)(JAGGED)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        assert not SOURCE.is_jagged()
        assert JAGGED.is_jagged()
        assert not RESULT.is_jagged()

        model = compile_model(
            [RESULT],
            detect_target(),
            "./tmp",
            f"test_jagged_to_padded_dense_{test_suffix}",
        )

        torch_offsets_dtype = string_to_torch_dtype(offsets_dtype)
        offsets_pt = {
            f"offsets{i}": torch.tensor(offsets, dtype=torch_offsets_dtype).cuda()
            for i, offsets in enumerate(offsets_list)
        }
        source_pt = get_random_torch_tensor(jagged_input_shape, dtype)
        result_pt = jagged_utils.jagged_to_dense(
            jagged=source_pt,
            offsets_list=list(offsets_pt.values()),
            dense_shape=jagged_max_shape,
            padding_value=padding_value,
        )
        result = torch.empty_like(result_pt)

        inputs = {"source": source_pt, **offsets_pt}
        model.run_with_tensors(inputs, [result])

        torch.testing.assert_close(result, result_pt)

    @parameterized.expand(
        [
            param(1, "int32", [4, 3, 8], "float16", 0.0),
            param(2, "int32", [4, 3, 4], "float16", 1e2),
            param(3, "int32", [4, 3, 2], "float16", 0.0),
            param(4, "int32", [4, 3, 1], "float16", 1e2),
            param(5, "int64", [4, 3, 4], "float32", 0.0),
            param(6, "int64", [4, 3, 2], "float32", 1e5),
            param(7, "int64", [4, 3, 1], "float32", 1e5),
        ]
    )
    def test_jagged_to_padded_dense_single_offsets(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dtype,
        padding_value,
    ):
        self._test_jagged_to_padded_dense(
            jagged_max_shape=jagged_max_shape,
            offsets_list=[[0, 1, 4, 6, 7]],
            dtype=dtype,
            offsets_dtype=offsets_dtype,
            test_suffix=f"single_offsets_{dtype}_{i}",
            padding_value=padding_value,
        )

    @parameterized.expand(
        [
            param(1, "int32", [3, 4, 5, 150, 3, 8], "float16", 0.0),
            param(2, "int32", [3, 4, 5, 150, 1, 4], "float16", 1e2),
            param(3, "int32", [3, 4, 5, 150, 3, 2], "float16", 0.0),
            param(4, "int32", [3, 4, 5, 150, 1, 1], "float16", 1e2),
            param(5, "int64", [3, 4, 5, 150, 1, 4], "float32", 0.0),
            param(6, "int64", [3, 4, 5, 150, 3, 2], "float32", 1e5),
            param(7, "int64", [3, 4, 5, 150, 3, 1], "float32", 1e5),
        ]
    )
    def test_jagged_to_padded_dense_multiple_offsets(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dtype,
        padding_value,
    ):
        self._test_jagged_to_padded_dense(
            jagged_max_shape=jagged_max_shape,
            offsets_list=[
                [0, 1, 3, 5],
                [0, 2, 4, 7, 9, 10],
                [0, 6, 8, 19, 23, 45, 67, 98, 123, 256, 321],
            ],
            dtype=dtype,
            offsets_dtype=offsets_dtype,
            test_suffix=f"multiple_offsets_{dtype}_{i}",
            padding_value=padding_value,
        )

    def _benchmark_jagged_to_padded_dense(
        self,
        B: int,
        N: int,
        D: int,
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        test_suffix: str = "",
        num_iters: int = 1000,
    ):
        batch_dim = IntVar(values=[1, B], name="batch_size")
        jagged_dim = JaggedDim(min_value=0, max_value=N)
        total_length_dim = IntVar(values=[1, B * N], name="total_length")
        embedding_dim = IntImm(value=D, name="embedding_dim")
        offsets_dim = IntVar(values=[2, B + 1], name="offsets_dim")

        SOURCE = Tensor(
            shape=[
                total_length_dim,
                embedding_dim,
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
            jagged_dims=[jagged_dim],
        )(SOURCE, OFFSETS_LIST)

        RESULT = ops.jagged_to_padded_dense()(JAGGED)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        model = compile_model(
            [RESULT],
            detect_target(),
            "./tmp",
            f"benchmark_jagged_to_padded_dense_{test_suffix}",
        )

        random.seed(0)
        load_factors = [i / 20 for i in range(1, 21)]
        offset_tensors = [
            jagged_utils.generate_offsets(
                batch_size=B,
                max_seq_len=N,
                load_factor=load_factor,
                offsets_dtype=offsets_dtype,
            )
            for load_factor in load_factors
        ]

        results = []
        for load_factor, offsets_pt in zip(load_factors, offset_tensors):
            total_length = offsets_pt[-1].item()
            source_pt = get_random_torch_tensor([total_length, D], dtype)
            inputs = {"source": source_pt, "offsets": offsets_pt}
            outputs = [
                torch.zeros(
                    (B, N, D), dtype=string_to_torch_dtype(dtype), device="cuda"
                )
            ]

            with tempfile.NamedTemporaryFile("r") as f:
                model.profile_with_tensors(
                    inputs=inputs,
                    outputs=outputs,
                    num_iters=num_iters,
                    filename=f.name,
                )
                profiling_data = json.loads(f.read())
                jagged_to_padded_dense_records = [
                    profiling_data[func_name]
                    for func_name in profiling_data
                    if func_name.startswith("jagged_to_padded_dense")
                ]
                assert len(jagged_to_padded_dense_records) == 1
                runtime_ms = jagged_to_padded_dense_records[0]["ms_per_iter"]

            jagged_item = total_length * D  # total items to read: the jagged volume
            dense_item = B * N * D  # total items to write: the dense volume
            size = 2 if dtype == "float16" else 4  # size of individual data value
            bandwidth = (
                (jagged_item + dense_item) * size / (runtime_ms * 1e-3 * 1e9)
            )  # GB/s
            results.append([load_factor, runtime_ms, bandwidth])

        print()
        print(f"{B=}, {N=}, {D=}, {dtype=}:")
        print()

        for load_factor, runtime_ms, bandwidth in results:
            print(
                f"load factor: {int(load_factor * 100)}%, "
                f"runtime: {round(runtime_ms, 6)} ms, "
                f"bandwidth: {round(bandwidth, 3)} GB/s"
            )

    def _test_benchmark_jagged_to_padded_dense(self):
        self._benchmark_jagged_to_padded_dense(
            B=1024,
            N=260,
            D=256,
            dtype="float16",
            offsets_dtype="int32",
            test_suffix="benchmark",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
