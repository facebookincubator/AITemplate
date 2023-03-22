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
import json
import random
import tempfile
import unittest
from typing import List

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import JaggedDim
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.jagged_utils import add_jagged_dense_ref, generate_offsets
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import param, parameterized


class JaggedElementwiseTestCase(unittest.TestCase):
    def _test_jagged_dense_elementwise_add(
        self,
        jagged_max_shape: List[int],
        offsets_list: List[List[int]],
        dense_shape: List[int],
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        use_jagged_space_indexing: bool = False,
        test_suffix: str = "",
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

        assert len(dense_shape) <= len(jagged_max_shape)
        dense_dims = [IntImm(dim) for dim in dense_shape]
        if len(dense_shape) == len(jagged_max_shape):
            assert dense_shape[0] == jagged_max_shape[0]
            dense_dims[0] = batch_dim

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
        DENSE = Tensor(
            shape=dense_dims,
            name="dense",
            dtype=dtype,
            is_input=True,
        )

        JAGGED = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=jagged_dims,
        )(SOURCE, OFFSETS_LIST)

        RESULT = ops.elementwise(FuncEnum.ADD)(JAGGED, DENSE)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        assert not SOURCE.is_jagged()
        assert not DENSE.is_jagged()
        assert JAGGED.is_jagged()
        assert RESULT.is_jagged()

        model = compile_model(
            [RESULT],
            detect_target(use_jagged_space_indexing=use_jagged_space_indexing),
            "./tmp",
            f"test_jagged_dense_elementwise_add_{test_suffix}",
        )

        torch_offsets_dtype = string_to_torch_dtype(offsets_dtype)
        offsets_pt = {
            f"offsets{i}": torch.tensor(offsets, dtype=torch_offsets_dtype).cuda()
            for i, offsets in enumerate(offsets_list)
        }
        source_pt = get_random_torch_tensor(jagged_input_shape, dtype)
        dense_pt = get_random_torch_tensor(dense_shape, dtype)
        result_pt = add_jagged_dense_ref(
            jagged=source_pt,
            offsets_list=list(offsets_pt.values()),
            jagged_max_shape=jagged_max_shape,
            dense=dense_pt,
        )
        result = torch.empty_like(result_pt)

        inputs = {"source": source_pt, "dense": dense_pt, **offsets_pt}
        model.run_with_tensors(inputs, [result])

        torch.testing.assert_close(result, result_pt)

    @parameterized.expand(
        [
            param(1, "int32", [4, 3, 4], [4, 3, 4]),
            param(2, "int32", [4, 3, 2], [4, 3, 1]),
            param(3, "int32", [4, 3, 1], [4, 3, 2]),
            param(4, "int32", [4, 3, 2], [4, 1, 1]),
            param(5, "int32", [4, 3, 2], [3, 1]),
            param(6, "int64", [4, 3, 1], [2]),
            param(7, "int64", [4, 3, 5, 6, 8], [4, 3, 5, 6, 8]),
            param(8, "int64", [4, 3, 1, 6, 1], [4, 3, 5, 1, 8]),
            param(9, "int64", [4, 3, 1, 6, 1], [4, 1, 1, 1, 1]),
            param(10, "int64", [4, 3, 1, 1, 2], [3, 5, 6, 2]),
        ]
    )
    def test_jagged_dense_elementise_add_single_offsets_fp16(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dense_shape,
    ):
        for use_jagged_space_indexing in [False, True]:
            self._test_jagged_dense_elementwise_add(
                jagged_max_shape=jagged_max_shape,
                offsets_list=[[0, 1, 4, 6, 7]],
                dense_shape=dense_shape,
                dtype="float16",
                offsets_dtype=offsets_dtype,
                use_jagged_space_indexing=use_jagged_space_indexing,
                test_suffix=f"single_offsets_fp16_{i}_{use_jagged_space_indexing}",
            )

    @parameterized.expand(
        [
            param(1, "int32", [3, 4, 5, 150, 3, 4], [3, 4, 5, 150, 3, 4]),
            param(2, "int32", [3, 4, 5, 150, 1, 4], [3, 4, 5, 150, 3, 1]),
            param(3, "int32", [3, 4, 5, 150, 3, 4], [1]),
            param(4, "int64", [3, 4, 5, 150, 1, 1], [150, 3, 4]),
            param(5, "int64", [3, 4, 5, 150, 3, 4], [3, 1, 1, 1, 1, 1]),
        ]
    )
    def test_jagged_dense_elementise_add_multiple_offsets_fp16(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dense_shape,
    ):
        for use_jagged_space_indexing in [False, True]:
            self._test_jagged_dense_elementwise_add(
                jagged_max_shape=jagged_max_shape,
                offsets_list=[
                    [0, 1, 3, 5],
                    [0, 2, 4, 7, 9, 10],
                    [0, 6, 8, 19, 23, 45, 67, 98, 123, 256, 321],
                ],
                dense_shape=dense_shape,
                dtype="float16",
                offsets_dtype=offsets_dtype,
                use_jagged_space_indexing=use_jagged_space_indexing,
                test_suffix=f"multiple_offsets_fp16_{i}_{use_jagged_space_indexing}",
            )

    @parameterized.expand(
        [
            param(1, "int32", [4, 3, 2], [4, 3, 2]),
            param(2, "int64", [4, 3, 5, 6, 7], [4, 3, 5, 6, 7]),
            param(3, "int64", [4, 3, 1, 1, 1], [3, 5, 6, 7]),
        ]
    )
    def test_jagged_dense_elementise_add_single_offsets_fp32(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dense_shape,
    ):
        self._test_jagged_dense_elementwise_add(
            jagged_max_shape=jagged_max_shape,
            offsets_list=[[0, 1, 4, 6, 7]],
            dense_shape=dense_shape,
            dtype="float32",
            offsets_dtype=offsets_dtype,
            use_jagged_space_indexing=False,
            test_suffix=f"single_offsets_fp32_{i}",
        )

    @parameterized.expand(
        [
            param(1, "int32", [3, 4, 5, 150, 3, 4], [3, 4, 5, 150, 3, 4]),
            param(2, "int64", [3, 4, 5, 150, 1, 1], [150, 3, 4]),
        ]
    )
    def test_jagged_dense_elementise_add_multiple_offsets_fp32(
        self,
        i,
        offsets_dtype,
        jagged_max_shape,
        dense_shape,
    ):
        self._test_jagged_dense_elementwise_add(
            jagged_max_shape=jagged_max_shape,
            offsets_list=[
                [0, 1, 3, 5],
                [0, 2, 4, 7, 9, 10],
                [0, 6, 8, 19, 23, 45, 67, 98, 123, 256, 321],
            ],
            dense_shape=dense_shape,
            dtype="float32",
            offsets_dtype=offsets_dtype,
            use_jagged_space_indexing=False,
            test_suffix=f"multiple_offsets_fp32_{i}",
        )

    def _test_jagged_jagged_elementwise_add(
        self,
        jagged_max_prefix_shape: List[int],
        jagged1_inner_shape: List[int],
        jagged2_inner_shape: List[int],
        implicit_jagged_input: bool,
        offsets_list: List[List[int]],
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        test_suffix: str = "",
    ):
        assert len(jagged1_inner_shape) == len(jagged2_inner_shape)

        batch_size = jagged_max_prefix_shape[0]
        batch_dim = IntVar(values=[1, batch_size * 2], name="batch_size")

        jagged_dims_max_values = jagged_max_prefix_shape[1 : 1 + len(offsets_list)]
        jagged_dims = [
            JaggedDim(min_value=0, max_value=max_value)
            for max_value in jagged_dims_max_values
        ]

        total_length = offsets_list[-1][-1]
        total_length_dim = IntVar(values=[1, total_length * 2], name="total_length")

        jagged1_inner_dims = [IntImm(dim) for dim in jagged1_inner_shape]
        jagged1_input_shape = [total_length] + jagged1_inner_shape
        jagged2_inner_dims = [IntImm(dim) for dim in jagged2_inner_shape]
        jagged2_input_shape = [total_length] + jagged2_inner_shape

        offsets_dims = [
            IntVar(values=[2, len(offsets) * 2]) for offsets in offsets_list
        ]

        SOURCE1 = Tensor(
            shape=[
                total_length_dim,
                *jagged1_inner_dims,
            ],
            name="source1",
            dtype=dtype,
            is_input=True,
        )
        SOURCE2 = Tensor(
            shape=[
                total_length_dim,
                *jagged2_inner_dims,
            ],
            name="source2",
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

        if implicit_jagged_input:
            JAGGED1 = SOURCE1
        else:
            JAGGED1 = ops.make_jagged(
                batch_dim=batch_dim,
                jagged_dims=jagged_dims,
            )(SOURCE1, OFFSETS_LIST)

        JAGGED2 = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=jagged_dims,
        )(SOURCE2, OFFSETS_LIST)

        RESULT = ops.elementwise(FuncEnum.ADD)(JAGGED1, JAGGED2)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        if implicit_jagged_input:
            # SOURCE1 is "converted" into a jagged Tensor
            # in the ops.elementwise by replacing its first
            # dim with the JaggedIntVar from JAGGED 2
            assert SOURCE1.is_jagged()
        else:
            assert not SOURCE1.is_jagged()
        assert not SOURCE2.is_jagged()
        assert JAGGED1.is_jagged()
        assert JAGGED2.is_jagged()
        assert RESULT.is_jagged()

        model = compile_model(
            [RESULT],
            detect_target(),
            "./tmp",
            f"test_jagged_jagged_elementwise_add_{test_suffix}",
        )

        torch_offsets_dtype = string_to_torch_dtype(offsets_dtype)
        offsets_pt = {
            f"offsets{i}": torch.tensor(offsets, dtype=torch_offsets_dtype).cuda()
            for i, offsets in enumerate(offsets_list)
        }
        source1_pt = get_random_torch_tensor(jagged1_input_shape, dtype)
        source2_pt = get_random_torch_tensor(jagged2_input_shape, dtype)
        result_pt = source1_pt + source2_pt  # jagged inputs are treated as dense
        result = torch.empty_like(result_pt)

        inputs = {"source1": source1_pt, "source2": source2_pt, **offsets_pt}
        model.run_with_tensors(inputs, [result])

        torch.testing.assert_close(result, result_pt)

    @parameterized.expand(
        [
            param(1, "int32", [4, 3], [5], [5], False),
            param(2, "int32", [4, 3], [5], [1], False),
            param(3, "int64", [4, 3], [1], [5], True),
            param(4, "int64", [4, 3], [5, 1, 7], [1, 6, 1], False),
            param(5, "int64", [4, 3], [5, 6, 7], [1, 6, 7], True),
        ]
    )
    def test_jagged_jagged_elementise_add_single_offsets_fp16(
        self,
        i,
        offsets_dtype,
        jagged_max_prefix_shape,
        jagged1_inner_shape,
        jagged2_inner_shape,
        implicit_jagged_input,
    ):
        self._test_jagged_jagged_elementwise_add(
            jagged_max_prefix_shape=jagged_max_prefix_shape,
            jagged1_inner_shape=jagged1_inner_shape,
            jagged2_inner_shape=jagged2_inner_shape,
            implicit_jagged_input=implicit_jagged_input,
            offsets_list=[[0, 1, 4, 6, 7]],
            dtype="float16",
            offsets_dtype=offsets_dtype,
            test_suffix=f"single_offsets_fp16_{i}",
        )

    @parameterized.expand(
        [
            param(1, "int32", [3, 4, 5, 200], [10], [10], False),
            param(2, "int32", [3, 4, 5, 200], [1, 2], [2, 1], True),
            param(3, "int64", [3, 4, 5, 150], [6, 7, 8], [6, 7, 8], False),
            param(4, "int64", [3, 4, 5, 150], [6, 1, 8], [1, 7, 1], True),
        ]
    )
    def test_jagged_jagged_elementise_add_multiple_offsets_fp16(
        self,
        i,
        offsets_dtype,
        jagged_max_prefix_shape,
        jagged1_inner_shape,
        jagged2_inner_shape,
        implicit_jagged_input,
    ):
        self._test_jagged_jagged_elementwise_add(
            jagged_max_prefix_shape=jagged_max_prefix_shape,
            jagged1_inner_shape=jagged1_inner_shape,
            jagged2_inner_shape=jagged2_inner_shape,
            implicit_jagged_input=implicit_jagged_input,
            offsets_list=[
                [0, 1, 3, 5],
                [0, 2, 4, 7, 9, 10],
                [0, 6, 8, 19, 23, 45, 67, 98, 123, 256, 321],
            ],
            dtype="float16",
            offsets_dtype=offsets_dtype,
            test_suffix=f"multiple_offsets_fp16_{i}",
        )

    def _benchmark_jagged_dense_elementwise_add(
        self,
        B: int,
        N: int,
        D: int,
        num_dense_inputs: int,
        dtype: str = "float16",
        offsets_dtype: str = "int32",
        use_jagged_space_indexing: bool = False,
        test_suffix: str = "",
        num_iters: int = 1000,
    ):
        batch_dim = IntVar(values=[1, B], name="batch_size")
        jagged_dim = JaggedDim(min_value=0, max_value=N)
        total_length_dim = IntVar(values=[1, B * N], name="total_length")
        sequence_dim = IntImm(value=N, name="sequence_dim")
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
        DENSE_INPUTS = [
            Tensor(
                shape=[
                    batch_dim,
                    sequence_dim,
                    embedding_dim,
                ],
                name=f"dense_{i}",
                dtype=dtype,
                is_input=True,
            )
            for i in range(num_dense_inputs)
        ]

        JAGGED = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=[jagged_dim],
        )(SOURCE, OFFSETS_LIST)

        RESULT = JAGGED
        for DENSE in DENSE_INPUTS:
            RESULT = ops.elementwise(FuncEnum.ADD)(RESULT, DENSE)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        model = compile_model(
            [RESULT],
            detect_target(use_jagged_space_indexing=use_jagged_space_indexing),
            "./tmp",
            f"benchmark_jagged_dense_elementwise_add_{test_suffix}",
        )

        random.seed(0)
        load_factors = [i / 20 for i in range(1, 21)]
        offset_tensors = [
            generate_offsets(
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
            dense_inputs_pt = {
                f"dense_{i}": get_random_torch_tensor([B, N, D], dtype)
                for i in range(num_dense_inputs)
            }
            source_pt = get_random_torch_tensor([total_length, D], dtype)
            inputs = {"source": source_pt, **dense_inputs_pt, "offsets": offsets_pt}
            outputs = [torch.empty_like(source_pt)]

            with tempfile.NamedTemporaryFile("r") as f:
                model.profile_with_tensors(
                    inputs=inputs,
                    outputs=outputs,
                    num_iters=num_iters,
                    filename=f.name,
                )
                profiling_data = json.loads(f.read())
                fused_elementwise_records = [
                    profiling_data[func_name]
                    for func_name in profiling_data
                    if func_name.startswith("fused_elementwise")
                ]
                assert len(fused_elementwise_records) == 1
                runtime_ms = fused_elementwise_records[0]["ms_per_iter"]

            items = total_length * D  # total items to read / write: the jagged volume
            size = 2 if dtype == "float16" else 4  # size of individual data value
            io_num = num_dense_inputs + 2  # num_dense_inputs + 1 inputs, 1 output
            bandwidth = io_num * items * size / (runtime_ms * 1e-3 * 1e9)  # GB/s
            results.append([load_factor, runtime_ms, bandwidth])

        print()
        print(f"{B=}, {N=}, {D=}, {num_dense_inputs=}, {dtype=}:")
        print()

        for load_factor, runtime_ms, bandwidth in results:
            print(
                f"load factor: {int(load_factor * 100)}%, "
                f"runtime: {round(runtime_ms, 6)} ms, "
                f"bandwidth: {round(bandwidth, 3)} GB/s"
            )

    def _test_benchmark_jagged_dense_elementise_add(self):
        self._benchmark_jagged_dense_elementwise_add(
            B=1024,
            N=260,
            D=256,
            num_dense_inputs=2,
            dtype="float16",
            offsets_dtype="int32",
            use_jagged_space_indexing=False,
            test_suffix="benchmark",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
