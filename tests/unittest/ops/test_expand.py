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
import math
import sys
import unittest

import torch

from aitemplate import compiler
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar, JaggedDim, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor, graph_has_op, has_op
from aitemplate.utils import graph_utils
from parameterized import param, parameterized


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ExpandTestCase(unittest.TestCase):
    def test_expand_fails_mismatched_ndim(self):
        x = Tensor(shape=[5, IntVar([1, 10]), 5])
        expand_shape = [5, -1]
        self.assertRaises(ValueError, ops.expand().__call__, x, expand_shape)

    def test_expand_fails_non_singleton_dim(self):
        x = Tensor(shape=[5, 1, 2])
        expand_shape = [6, 1, 2]
        self.assertRaises(ValueError, ops.expand().__call__, x, expand_shape)

        x = Tensor(shape=[IntVar([1, 10])])
        expand_shape = [20]
        self.assertRaises(ValueError, ops.expand().__call__, x, expand_shape)

    def _test_no_op_expands_removed_static_shapes(
        self,
        test_name="no_op_expands_removed_static_shapes",
        dtype="float16",
    ):
        x = Tensor(
            [1, 2, 3],
            name="input_0",
            is_input=True,
            dtype=dtype,
        )
        y = ops.expand()(x, [1, -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(y, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        z_pt = x_pt * x_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(z, detect_target(), "./tmp", test_name) as module:
            module.run_with_tensors({"input_0": x_pt}, {"output_0": z_ait})
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_static_shapes_fp16(self):
        self._test_no_op_expands_removed_static_shapes(
            test_name="no_op_expands_removed_static_shapes_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_no_op_expands_removed_static_shapes_fp32(self):
        self._test_no_op_expands_removed_static_shapes(
            test_name="no_op_expands_removed_static_shapes_fp32",
            dtype="float32",
        )

    def _test_no_op_expands_removed_dynamic_shapes(
        self,
        test_name="no_op_expands_removed_dynamic_shapes",
        dtype="float16",
    ):
        dynamic_dim = IntVar([1, 5], name="dynamic_dim")
        x = Tensor(
            [1, dynamic_dim, 3],
            name="input_0",
            is_input=True,
            dtype=dtype,
        )
        y = ops.expand()(x, [IntVar([1, 1]), -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(y, y)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        z_pt = x_pt * x_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(z, detect_target(), "./tmp", test_name) as module:
            module.run_with_tensors({"input_0": x_pt}, {"output_0": z_ait})
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_dynamic_shapes_fp16(self):
        self._test_no_op_expands_removed_dynamic_shapes(
            test_name="no_op_expands_removed_dynamic_shapes_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_no_op_expands_removed_dynamic_shapes_fp32(self):
        self._test_no_op_expands_removed_dynamic_shapes(
            test_name="no_op_expands_removed_dynamic_shapes_fp32",
            dtype="float32",
        )

    def _test_no_op_expands_removed_size_op(
        self,
        test_name="no_op_expands_removed_size_op",
        dtype="float16",
    ):
        x = Tensor(
            [1, 2, 3],
            name="input_0",
            is_input=True,
            dtype=dtype,
        )
        y = Tensor(
            [IntVar([1, 1]), 2, 3],
            name="input_1",
            is_input=True,
            dtype=dtype,
        )
        x_size = ops.size()(x, 0)
        y_size = ops.size()(y, 0)
        x_expand = ops.expand()(x, [x_size, -1, -1])
        y_expand = ops.expand()(y, [y_size, -1, -1])
        z = ops.elementwise(FuncEnum.MUL)(x_expand, y_expand)
        z._attrs["is_output"] = True
        z._attrs["name"] = "output_0"

        x_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        y_pt = get_random_torch_tensor([1, 2, 3], dtype=dtype)
        z_pt = x_pt * y_pt
        z_ait = torch.empty_like(z_pt)
        with compile_model(z, detect_target(), "./tmp", test_name) as module:
            module.run_with_tensors(
                {"input_0": x_pt, "input_1": y_pt}, {"output_0": z_ait}
            )
            self.assertFalse(graph_has_op(module.debug_sorted_graph, "expand"))
            self.assertTrue(torch.equal(z_ait, z_pt))

    def test_no_op_expands_removed_size_op_fp16(self):
        self._test_no_op_expands_removed_size_op(
            test_name="no_op_expands_removed_size_op_fp16",
            dtype="float16",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_no_op_expands_removed_size_op_fp32(self):
        self._test_no_op_expands_removed_size_op(
            test_name="no_op_expands_removed_size_op_fp32",
            dtype="float32",
        )

    def test_no_op_expand_elementwise_jagged_dense_inputs(self):
        total_length = IntVar([1, 100])
        batch_dim = IntVar([1, 10])
        offsets_dim = IntVar([2, 11])
        embedding_dim = IntImm(128)
        max_seq_len = 10

        X = Tensor(
            [batch_dim, 1, embedding_dim],
            name="x",
            is_input=True,
            dtype="float16",
        )
        SOURCE = Tensor(
            [total_length, embedding_dim],
            name="source",
            is_input=True,
            dtype="float16",
        )
        OFFSETS_LIST = [
            Tensor(
                shape=[offsets_dim],
                name="offsets",
                is_input=True,
                dtype="int32",
            )
        ]

        JAGGED = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=[
                JaggedDim(0, max_seq_len),
            ],
        )(
            source=SOURCE,
            offsets_list=OFFSETS_LIST,
        )

        Y = ops.expand()(X, [batch_dim, max_seq_len, -1])
        Z = ops.elementwise(FuncEnum.MUL)(JAGGED, Y)

        graph = compiler.transform.toposort([Z])
        compiler.transform.remove_no_ops(graph)
        sorted_ops = graph_utils.get_sorted_ops(graph)

        assert not has_op(sorted_ops, "expand")

    @parameterized.expand(
        [
            param("fp32_small_noadd_1", "float32", [10, 1, 5], [-1, 10, 5]),
            param("fp32_small_noadd_2", "float32", [10, 1, 8], [-1, 10, 8]),
            param("fp32_small_noadd_3", "float32", [10, 1, 2], [-1, 10, 2]),
            param("fp32_small_noadd_4", "float32", [10, 1, 5], [10, 10, 5]),
            param("fp32_small_1", "float32", [10, 1, 5], [3, 10, 10, 5]),
            param("fp32_small_2", "float32", [3, 1, 5], [3, 3, 3, -1]),
            param("fp32_small_3", "float32", [2, 1, 4, 1, 6], [-1, 10, 4, 5, 6]),
            param("fp32_small_var_1", "float32", [10, 1, 5], [3, 10, 10, 5], False),
            param("fp32_small_var_2", "float32", [1, 1, 5], [3, 3, 10, -1], False),
            param(
                "fp32_small_var_3", "float32", [2, 1, 4, 1, 6], [-1, 10, 4, 5, 6], False
            ),
            param("float16_small_1", "float16", [2, 3, 1, 5], [2, -1, 3, 10, 5]),
            param("float16_small_2", "float16", [1, 2, 10], [10, 2, 10]),
            param("bfloat16_small_1", "bfloat16", [2, 3, 1, 5], [2, -1, 3, 10, 5]),
            param("int64_small_1", "int64", [2, 3, 1, 5], [2, -1, 3, 10, 5]),
            param(
                "fp32_large_1",
                "float32",
                [100, 1, 9, 3],
                [2, 20, -1, 100, 9, -1],
                "int32",
            ),
            param(
                "fp32_large_2",
                "float32",
                [101, 1, 91, 3],
                [-1, 100, 91, -1],
                "int64",
            ),
            param(
                "fp32_large_3",
                "float32",
                [100, 1, 9, 3],
                [2, 20, -1, 100, 9, -1],
                "int64",
            ),
            # Largest tests commented out, as these lead to GPU OOM failures on Github CircleCI Hardware
            # param(
            #    "fp32_large_4",
            #    "float32",
            #    [100, 1, 91, 3],
            #    [2, 20, -1, 100, 91, -1],
            #    "int64",
            # ),
            # param(
            #     "fp32_large_5",
            #     "float32",
            #     [101, 1, 91, 7],
            #     [3, 21, -1, 103, 91, -1],
            #     "int64",
            # ),
            # param(
            #     "fp32_large_repeat",
            #     "float32",
            #     [101, 1, 91, 8],
            #     [1000, -1, -1, -1, -1],
            #     "int64",
            # ),
            # param(
            #     "fp32_large_var_2",
            #     "float32",
            #     [100, 1, 9, 3],
            #     [2, 20, -1, 100, 9, -1],
            #     False,
            #     "int64",
            # ),
            # param(
            #     "benchmark_1",
            #     "float32",
            #     [100, 1, 9, 4],
            #     [20, 20, 100, 100, 9, -1],
            #     True,
            #     "int64",
            # ),
            # param(
            #     "benchmark_2",
            #     "int64",
            #     [100, 1, 9, 4],
            #     [20, 20, 100, 100, 9, -1],
            #     True,
            #     "int64",
            # ),
            # param(
            #     "benchmark_3",
            #     "float16",
            #     [100, 1, 9, 4],
            #     [20, 20, 100, 100, 9, -1],
            #     True,
            #     "int64",
            # ),
            param(
                "benchmark_var_1",
                "float32",
                [100, 1, 9, 4],
                [20, 20, 100, 100, 9, -1],
                False,
                "int64",
            ),
            # param(
            #     "benchmark_var_2",
            #     "int64",
            #     [100, 1, 9, 4],
            #     [20, 20, 100, 100, 9, -1],
            #     False,
            #     "int64",
            # ),
            # param(
            #     "benchmark_var_3",
            #     "float16",
            #     [100, 1, 9, 4],
            #     [20, 20, 100, 100, 9, -1],
            #     False,
            #     "int64",
            # ),
            param("fp32_m_1", "float32", [5, 1, 3, 2], [2, 2, -1, 5, 3, -1]),
            param("fp32_m_2", "float32", [5, 1, 3, 5], [2, 2, -1, 5, 3, -1]),
            param("edge_case_shapes_1", "float32", [1, 1, 1, 1], [1, 1, -1, 1, -1, 1]),
            param("edge_case_shapes_2", "float32", [1], [-1]),
            param("edge_case_shapes_3", "float32", [3], [-1]),
            param("edge_case_shapes_4", "float32", [1], [1]),
            param("edge_case_shapes_5", "float32", [1, 1], [1, 0]),
            param("edge_case_shapes_6", "float32", [2, 0], [-1, -1]),
            param("edge_case_shapes_7", "float32", [2, 0], [2, 0]),
            param(
                "edge_case_shapes_var_1",
                "float32",
                [1, 1, 1, 1],
                [1, 1, -1, 1, -1, 1],
                False,
            ),
            param("edge_case_shapes_var_2", "float32", [1], [-1], False),
            param("edge_case_shapes_var_3", "float32", [3], [-1], False),
            param("edge_case_shapes_var_4", "float32", [1], [1], False),
            param("edge_case_shapes_var_5", "float32", [1, 1], [1, 0], False),
            param("edge_case_shapes_var_6", "float32", [2, 0], [-1, -1], False),
            param("edge_case_shapes_var_6", "float32", [2, 0], [2, 0], False),
        ]
    )
    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_expand_op(
        self,
        name,
        dtype,
        src_shape,
        expand_shape,
        optimize_fixed_dims=True,
        index_type="int64",
    ):
        x = Tensor(
            src_shape,
            name="X",
            is_input=True,
            dtype=dtype,
        )
        y = ops.expand()(
            x,
            expand_shape,
            optimize_fixed_dims=optimize_fixed_dims,
            index_type=index_type,
        )
        y._attrs["is_output"] = True
        y._attrs["name"] = "Y"
        if dtype != "int64":
            x_pt = get_random_torch_tensor(src_shape, dtype=dtype)
        else:
            x_pt = torch.arange(
                1, math.prod(src_shape) + 1, 1, dtype=torch.int64, device="cuda"
            ).view(src_shape)
        y_pt = x_pt.expand(expand_shape)
        y_ait = torch.zeros_like(y_pt)
        stream = torch.cuda.default_stream()
        start_event_pt = torch.cuda.Event(enable_timing=True)
        end_event_pt = torch.cuda.Event(enable_timing=True)
        num_iters = 20
        with compile_model(
            y, detect_target(), "./tmp", "test_expand_codegen_" + name
        ) as module:
            module.run_with_tensors({"X": x_pt}, {"Y": y_ait})
            self.assertTrue(graph_has_op(module.debug_sorted_graph, "expand"))
            time_mean_ms, time_std_ms, result_tensors = module.benchmark_with_tensors(
                {"X": x_pt}, {"Y": y_ait}, count=num_iters
            )
        print(
            f"Write GB/sec:{1000*y_pt.numel()*y_pt.element_size()/time_mean_ms/(1024*1024*1024)}"
        )
        self.assertTrue(torch.equal(y_ait, y_pt))
        # measure time against torch.contiguous()
        cache_trasher = torch.zeros(1000, 1000, 42, device="cuda", requires_grad=False)
        sum_elapsed_pt = 0.0
        for _ in range(num_iters):
            # trash the L2 cache, just like the benchmark code of AIT does
            cache_trasher.normal_()
            start_event_pt = torch.cuda.Event(enable_timing=True)
            end_event_pt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event_pt.record(stream=stream)
            _ = y_pt.contiguous()
            end_event_pt.record(stream=stream)
            torch.cuda.synchronize()
            sum_elapsed_pt += start_event_pt.elapsed_time(end_event_pt)

        pt_time = sum_elapsed_pt / num_iters
        ait_throughput_write = (
            1000
            * y_pt.numel()
            * y_pt.element_size()
            / time_mean_ms
            / (1024 * 1024 * 1024)
        )
        ait_throughput_read_once = (
            1000
            * x_pt.numel()
            * x_pt.element_size()
            / time_mean_ms
            / (1024 * 1024 * 1024)
        )
        ait_throughput_total_lower_bound = (
            ait_throughput_write + ait_throughput_read_once
        )  # Assuming we just read the input once
        ait_throughput_total_upper_bound = (
            ait_throughput_write * 2
        )  # Assuming every byte written has been read as well

        pt_throughput_write = (
            1000 * y_pt.numel() * y_pt.element_size() / pt_time / (1024 * 1024 * 1024)
        )
        pt_throughput_read = (  # Assuming we just read the input once
            1000 * x_pt.numel() * x_pt.element_size() / pt_time / (1024 * 1024 * 1024)
        )

        pt_throughput_total_lower_bound = (
            pt_throughput_write + pt_throughput_read
        )  # Assuming we just read the input once
        pt_throughput_total_upper_bound = (
            pt_throughput_write * 2
        )  # Assuming every byte written has been read as well

        # ait_speedup_percent = round(100.0 * pt_time / time_mean_ms - 100.0)
        ait_speedup_factor = f"{pt_time/time_mean_ms:.2f}"
        ait_expand_variant = "general"
        if optimize_fixed_dims:
            ait_expand_variant = "optimized"
        print(
            f"""Benchmark Summary (test_expand_op:{name}) - {src_shape} => {expand_shape}: dtype={dtype}, variant={ait_expand_variant}. AIT speedup={ait_speedup_factor}x. Throughputs in GB/sec.: Write: pt={pt_throughput_write:.1f}, ait={ait_throughput_write:.1f}, Total (lower): pt={pt_throughput_total_lower_bound:.1f}, ait={ait_throughput_total_lower_bound:.1f} Total (upper): pt={pt_throughput_total_upper_bound:.1f}, ait=={ait_throughput_total_upper_bound:.1f} ]
Benchmark note: Total throughput (lower) assumes the input is read once, Total throughput (upper) assumes every byte written has been read as well. The truth is inbetween due to caching of repeated reads.""",
            file=sys.stdout,
            flush=True,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
