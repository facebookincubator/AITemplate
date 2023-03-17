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
Unittests for the padded_dense_to_jagged op.
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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import param, parameterized


class PaddedDenseToJaggedTestCase(unittest.TestCase):
    def _test_padded_dense_to_jagged(
        self,
        jagged_max_shape: List[int],
        offsets_list: List[List[int]],
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        use_jagged_space_indexing: bool = False,
        test_suffix: str = "",
    ):
        batch_size = jagged_max_shape[0]
        batch_dim = IntVar(values=[1, batch_size * 2], name="batch_size")
        sequence_shape = jagged_max_shape[1 : 1 + len(offsets_list)]
        sequence_dims = [IntImm(value=dim) for dim in sequence_shape]
        inner_shape = jagged_max_shape[1 + len(offsets_list) :]
        inner_dims = [IntImm(value=dim) for dim in inner_shape]

        total_length = offsets_list[-1][-1]
        total_length_dim = IntVar(values=[1, total_length * 2], name="total_length")
        jagged_dims = [JaggedDim(min_value=0, max_value=N) for N in sequence_shape]

        offsets_dims = [
            IntVar(values=[2, len(offsets) * 2]) for offsets in offsets_list
        ]

        DENSE = Tensor(
            shape=[
                batch_dim,
                *sequence_dims,
                *inner_dims,
            ],
            name="dense",
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
        SOURCE = Tensor(
            shape=[
                total_length_dim,
                *inner_dims,
            ],
            name="source",
            dtype=dtype,
            is_input=True,
        )

        JAGGED = ops.padded_dense_to_jagged(total_length=total_length_dim)(
            x=DENSE,
            offsets_list=OFFSETS_LIST,
        )
        ANOTHER = ops.make_jagged(batch_dim=batch_dim, jagged_dims=jagged_dims)(
            source=SOURCE,
            offsets_list=OFFSETS_LIST,
        )

        RESULT = ops.elementwise(FuncEnum.ADD)(JAGGED, ANOTHER)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        assert not DENSE.is_jagged()
        assert JAGGED.is_jagged()
        assert ANOTHER.is_jagged()
        assert RESULT.is_jagged()

        model = compile_model(
            [RESULT],
            detect_target(use_jagged_space_indexing=use_jagged_space_indexing),
            "./tmp",
            f"test_padded_dense_to_jagged_{test_suffix}",
        )

        torch_offsets_dtype = string_to_torch_dtype(offsets_dtype)
        offsets_pt = {
            f"offsets{i}": torch.tensor(offsets, dtype=torch_offsets_dtype).cuda()
            for i, offsets in enumerate(offsets_list)
        }
        dense_pt = get_random_torch_tensor(jagged_max_shape, dtype)
        result_pt = jagged_utils.dense_to_jagged(
            dense=dense_pt,
            offsets_list=list(offsets_pt.values()),
        )

        source = torch.zeros_like(result_pt)
        result = torch.empty_like(result_pt)

        inputs = {"dense": dense_pt, "source": source, **offsets_pt}
        model.run_with_tensors(inputs, [result])

        torch.testing.assert_close(result, result_pt)

    @parameterized.expand(
        [
            param(1, "int32", [4, 3, 8], "float16"),
            param(2, "int32", [4, 3, 4], "float16"),
            param(3, "int32", [4, 3, 2], "float16"),
            param(4, "int32", [4, 3, 1], "float16"),
            param(5, "int64", [4, 3, 4], "float32"),
            param(6, "int64", [4, 3, 2], "float32"),
            param(7, "int64", [4, 3, 1], "float32"),
        ]
    )
    def test_padded_dense_to_jagged_single_offsets(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dtype,
    ):
        for use_jagged_space_indexing in [False, True]:
            self._test_padded_dense_to_jagged(
                jagged_max_shape=jagged_max_shape,
                offsets_list=[[0, 1, 4, 6, 7]],
                dtype=dtype,
                offsets_dtype=offsets_dtype,
                use_jagged_space_indexing=use_jagged_space_indexing,
                test_suffix=f"single_offsets_{dtype}_{i}",
            )

    @parameterized.expand(
        [
            param(1, "int32", [3, 4, 5, 150, 3, 8], "float16"),
            param(2, "int32", [3, 4, 5, 150, 1, 4], "float16"),
            param(3, "int32", [3, 4, 5, 150, 3, 2], "float16"),
            param(4, "int32", [3, 4, 5, 150, 1, 1], "float16"),
            param(5, "int64", [3, 4, 5, 150, 1, 4], "float32"),
            param(6, "int64", [3, 4, 5, 150, 3, 2], "float32"),
            param(7, "int64", [3, 4, 5, 150, 3, 1], "float32"),
        ]
    )
    def test_padded_dense_to_jagged_multiple_offsets(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dtype,
    ):
        for use_jagged_space_indexing in [False, True]:
            self._test_padded_dense_to_jagged(
                jagged_max_shape=jagged_max_shape,
                offsets_list=[
                    [0, 1, 3, 5],
                    [0, 2, 4, 7, 9, 10],
                    [0, 6, 8, 19, 23, 45, 67, 98, 123, 256, 321],
                ],
                dtype=dtype,
                offsets_dtype=offsets_dtype,
                use_jagged_space_indexing=use_jagged_space_indexing,
                test_suffix=f"multiple_offsets_{dtype}_{i}",
            )

    def _benchmark_padded_dense_to_jagged(
        self,
        B: int,
        N: int,
        D: int,
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        use_jagged_space_indexing: bool = False,
        test_suffix: str = "",
        num_iters: int = 1000,
    ):
        batch_dim = IntVar(values=[1, B], name="batch_size")
        sequence_dim = IntImm(value=N, name="sequence_dim")
        total_length_dim = IntVar(values=[1, B * N], name="total_length")
        embedding_dim = IntImm(value=D, name="embedding_dim")
        offsets_dim = IntVar(values=[2, B + 1], name="offsets_dim")

        DENSE = Tensor(
            shape=[
                batch_dim,
                sequence_dim,
                embedding_dim,
            ],
            name="dense",
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

        JAGGED = ops.padded_dense_to_jagged(total_length=total_length_dim)(
            x=DENSE,
            offsets_list=OFFSETS_LIST,
        )

        SOURCE = Tensor(
            shape=[
                total_length_dim,
                embedding_dim,
            ],
            name="source",
            dtype=dtype,
            is_input=True,
        )
        ANOTHER = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=[JaggedDim(min_value=0, max_value=N)],
        )(
            source=SOURCE,
            offsets_list=OFFSETS_LIST,
        )

        RESULT = ops.elementwise(FuncEnum.ADD)(JAGGED, ANOTHER)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        model = compile_model(
            [RESULT],
            detect_target(use_jagged_space_indexing=use_jagged_space_indexing),
            "./tmp",
            f"benchmark_padded_dense_to_jagged_{test_suffix}",
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
            dense_pt = get_random_torch_tensor([B, N, D], dtype)
            inputs = {"dense": dense_pt, "offsets": offsets_pt}
            outputs = [get_torch_empty_tensor([total_length, D], dtype)]
            source_pt = get_random_torch_tensor([total_length, D], dtype)
            inputs["source"] = source_pt

            with tempfile.NamedTemporaryFile("r") as f:
                model.profile_with_tensors(
                    inputs=inputs,
                    outputs=outputs,
                    num_iters=num_iters,
                    filename=f.name,
                )
                profiling_data = json.loads(f.read())
                padded_dense_to_jagged_records = [
                    profiling_data[func_name]
                    for func_name in profiling_data
                    if func_name.startswith("padded_dense_to_jagged")
                ]
                assert len(padded_dense_to_jagged_records) == 1
                runtime_ms = padded_dense_to_jagged_records[0]["ms_per_iter"]

            dense_item = total_length * D  # total items to read: the jagged volume
            jagged_item = total_length * D  # total items to read: the jagged volume
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

    def _test_benchmark_padded_dense_to_jagged(self):
        self._benchmark_padded_dense_to_jagged(
            B=1024,
            N=260,
            D=256,
            dtype="float16",
            offsets_dtype="int32",
            use_jagged_space_indexing=False,
            isolated_total_length=True,
            test_suffix="benchmark",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
