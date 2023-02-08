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
import unittest
from functools import partial

from typing import Callable, List, Tuple

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, test_utils
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils

from parameterized import param, parameterized


def _gen_simple_strided_ops(
    batch_dim: IntVar, n1: int, n2: int
) -> List[Tuple[str, Tensor, str, Callable[[torch.Tensor], torch.Tensor]]]:
    test_cases = [
        (
            "tanh",
            ops.elementwise(FuncEnum.TANH)(
                test_utils.gen_input_tensor([batch_dim, n1, n2], dtype="float16")
            ),
            "float16",
            torch.tanh,
        ),
        (
            "layernorm",
            ops.layernorm(normalized_shape=[IntImm(n2)])(
                test_utils.gen_input_tensor([batch_dim, n1, n2], dtype="float16")
            ),
            "float16",
            partial(torch.nn.functional.layer_norm, normalized_shape=[n2]),
        ),
        (
            "sum",
            ops.reduce_sum(2, keepdim=True)(
                test_utils.gen_input_tensor([batch_dim, n1, n2], dtype="float16")
            ),
            "float16",
            partial(torch.sum, dim=2, keepdim=True),
        ),
    ]
    target = detect_target()
    if target.name() == "cuda":
        test_cases.append(
            (
                "tanh",
                ops.elementwise(FuncEnum.TANH)(
                    test_utils.gen_input_tensor([batch_dim, n1, n2], dtype="float")
                ),
                "float",
                torch.tanh,
            )
        )
        test_cases.append(
            (
                "sum",
                ops.reduce_sum(2, keepdim=True)(
                    test_utils.gen_input_tensor([batch_dim, n1, n2], dtype="float")
                ),
                "float",
                partial(torch.sum, dim=2, keepdim=True),
            )
        )
    return test_cases


def _gen_fusible_view_ops_after_strided_op() -> List[
    Tuple[str, Callable[[Tensor], Tensor], str]
]:
    def reshape_op(input_tensor: Tensor):
        shape = input_tensor._attrs["shape"]
        return ops.reshape()(
            input_tensor,
            [-1, shape[1].value() * shape[2].value()],
        )

    def flatten_op(input_tensor: Tensor):
        return ops.flatten(start_dim=1, end_dim=-1)(input_tensor)

    test_cases = [
        ("reshape", reshape_op, "float16"),
        ("flatten", flatten_op, "float16"),
    ]
    target = detect_target()
    if target.name() == "cuda" and int(target._arch) >= 80:
        test_cases.append(("reshape", reshape_op, "float"))
    return test_cases


def _gen_non_fusible_view_ops_after_strided_op() -> List[
    Tuple[str, Callable[[Tensor], Tensor], str]
]:
    def reshape_op(input_tensor: Tensor):
        n2 = input_tensor._attrs["shape"][2].value()
        return ops.reshape()(input_tensor, [-1, n2])

    def flatten_op(input_tensor: Tensor):
        return ops.flatten(start_dim=0, end_dim=1)(input_tensor)

    test_cases = [
        ("reshape", reshape_op, "float16"),
        ("flatten", flatten_op, "float16"),
    ]
    target = detect_target()
    if target.name() == "cuda":
        test_cases.append(("flatten", flatten_op, "float"))
    return test_cases


def _gen_multiple_fusible_view_ops_after_strided_op() -> List[
    Tuple[str, Callable[[Tensor], Tensor], str]
]:
    def _get_shape(input_tensor: Tensor):
        return (
            input_tensor._attrs["shape"][1].value(),
            input_tensor._attrs["shape"][2].value(),
        )

    def multi_reshape(input_tensor: Tensor):
        n1, n2 = _get_shape(input_tensor)
        return ops.reshape()(
            ops.reshape()(
                input_tensor,
                [-1, n1 * n2],
            ),
            [-1, n1, n2],
        )

    def squeeze_unsqueeze(input_tensor: Tensor):
        n1, n2 = _get_shape(input_tensor)
        return ops.squeeze(dim=1)(ops.unsqueeze(dim=1)(input_tensor))

    test_cases = [
        ("multi_reshape", multi_reshape, "float16"),
        ("squeeze_unsqueeze", squeeze_unsqueeze, "float16"),
    ]
    target = detect_target()
    if target.name() == "cuda" and int(target._arch) >= 80:
        test_cases.append(("multi_reshape", multi_reshape, "float"))
    return test_cases


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_{param.args[0]}_{param.args[2]}"


class StridedViewOpTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(StridedViewOpTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by cuda sm<80",
    )
    @parameterized.expand(
        [
            param(f"single_gemm_{name}_fusion_{dtype}", func, dtype)
            for (name, func, dtype) in _gen_fusible_view_ops_after_strided_op()
        ],
        name_func=custom_name_func,
    )
    def test_single_gemm_and_view_fusible(self, test_name, func, dtype):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        K = 10
        input0 = test_utils.gen_input_tensor([batch_dim, N1, K], dtype=dtype)
        input1 = test_utils.gen_input_tensor([N2, K], dtype=dtype)
        X0 = ops.gemm_rcr()(input0, input1)
        Y = ops.elementwise(FuncEnum.TANH)(func(X0))
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input0_pt = get_random_torch_tensor([batch_size, N1, K], dtype)
            input1_pt = get_random_torch_tensor([N2, K], dtype)

            # Run PyTorch baseline.
            x0_pt = torch.matmul(input0_pt, input1_pt.transpose(0, 1))
            dim_to_value_dict = {"batch_size": batch_size}
            y_pt = torch.tanh(
                torch.reshape(
                    x0_pt, test_utils.get_shape(Y._attrs["shape"], dim_to_value_dict)
                )
            )
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors([input0_pt, input1_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
            self._test_id += 1

    @parameterized.expand(
        [
            param(f"single_bmm_{name}_fusion_{dtype}", func, dtype)
            for (
                name,
                func,
                dtype,
            ) in _gen_multiple_fusible_view_ops_after_strided_op()
        ],
        name_func=custom_name_func,
    )
    def test_single_bmm_and_multi_view_fusible(self, test_name, func, dtype):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        K = 10
        input0 = test_utils.gen_input_tensor([batch_dim, N1, K], dtype)
        input1 = test_utils.gen_input_tensor([batch_dim, K, N2], dtype)
        X0 = ops.bmm_rrr()(input0, input1)
        Y = ops.elementwise(FuncEnum.COS)(func(X0))
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input0_pt = get_random_torch_tensor([batch_size, N1, K], dtype)
            input1_pt = get_random_torch_tensor([batch_size, K, N2], dtype)

            # Run PyTorch baseline.
            x0_pt = torch.matmul(input0_pt, input1_pt)
            dim_to_value_dict = {"batch_size": batch_size}
            y_pt = torch.cos(
                torch.reshape(
                    x0_pt, test_utils.get_shape(Y._attrs["shape"], dim_to_value_dict)
                )
            )
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors([input0_pt, input1_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
            self._test_id += 1

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by cuda sm<80",
    )
    @parameterized.expand(
        [
            param(
                f"single_{op_name}_reshape_fusion_{dtype}",
                input_tensor,
                dtype,
                torch_func,
            )
            for (op_name, input_tensor, dtype, torch_func) in _gen_simple_strided_ops(
                IntVar([1, 128, 256], "batch_size"), n1=10, n2=8
            )
        ],
        name_func=custom_name_func,
    )
    def test_single_op_and_view_fusible(
        self, test_name, input_tensor, dtype, torch_func
    ):
        src_input = test_utils.get_src_input(input_tensor)
        batch_dim = src_input._attrs["shape"][0]
        n1 = src_input._attrs["shape"][1].value()
        n2 = src_input._attrs["shape"][2].value()
        Y = ops.elementwise(FuncEnum.TANH)(ops.reshape()(input_tensor, [batch_dim, -1]))
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input_pt = get_random_torch_tensor([batch_size, n1, n2], dtype)

            # Run PyTorch baseline.
            y_pt = torch.tanh(torch.reshape(torch_func(input_pt), [batch_size, -1]))
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors([input_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
            self._test_id += 1

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by cuda sm<80",
    )
    @parameterized.expand(
        [
            param(f"single_op_{name}_non_fusion_{dtype}", func, dtype)
            for (name, func, dtype) in _gen_non_fusible_view_ops_after_strided_op()
        ],
        name_func=custom_name_func,
    )
    def test_single_op_and_view_non_fusible(self, test_name, func, dtype):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        X0 = test_utils.gen_input_tensor([batch_dim, N1, N2], dtype=dtype)
        X1 = ops.elementwise(FuncEnum.TANH)(X0)
        Y = ops.elementwise(FuncEnum.TANH)(func(X1))
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        dll_name = f"test_{self._test_id}.so"
        module = compile_model([Y], target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            x0_pt = get_random_torch_tensor([batch_size, N1, N2], dtype)

            # Run PyTorch baseline.
            y_pt = torch.tanh(torch.reshape(torch.tanh(x0_pt), [-1, N2]))
            y = get_torch_empty_tensor(y_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors([x0_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))
            self._test_id += 1

    def _test_two_serial_view_outputs(self, dtype="float16"):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        X0 = test_utils.gen_input_tensor([batch_dim, N1, N2], dtype)
        X1 = ops.elementwise(FuncEnum.TANH)(X0)
        Y1 = ops.reshape()(X1, [-1, N1 * N2])
        Y2 = ops.reshape()(Y1, [-1, N1, N2])
        Y1._attrs["name"] = "output1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "output2"
        Y2._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model([Y1, Y2], target, "./tmp", f"two_view_outputs_{dtype}")

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input_pt = get_random_torch_tensor([batch_size, N1, N2], dtype)

            # Run PyTorch baseline.
            y1_pt = torch.reshape(torch.tanh(input_pt), [batch_size, N1 * N2])
            y2_pt = torch.reshape(y1_pt, [batch_size, N1, N2])
            y1 = get_torch_empty_tensor(y1_pt.shape, dtype)
            y2 = get_torch_empty_tensor(y2_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors([input_pt], [y1, y2])

            # Do comparisons.
            self.assertTrue(torch.allclose(y1, y1_pt, atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(y2, y2_pt, atol=1e-2, rtol=1e-2))

    def _test_two_parallel_views(self, dtype="float16"):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        X0 = test_utils.gen_input_tensor([batch_dim, N1, N2], dtype)
        X1 = ops.elementwise(FuncEnum.TANH)(X0)
        Y1 = ops.elementwise(FuncEnum.TANH)(ops.reshape()(X1, [-1, N1 * N2]))
        Y2 = ops.elementwise(FuncEnum.TANH)(ops.reshape()(X1, [-1, N1, N2]))
        Y1._attrs["name"] = "output1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "output2"
        Y2._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(
            [Y1, Y2], target, "./tmp", f"two_parallel_view_outputs_{dtype}"
        )

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input_pt = get_random_torch_tensor([batch_size, N1, N2], dtype)
            x1_pt = torch.tanh(input_pt)

            # Run PyTorch baseline.
            y1_pt = torch.tanh(torch.reshape(x1_pt, [batch_size, N1 * N2]))
            y2_pt = torch.tanh(torch.reshape(x1_pt, [batch_size, N1, N2]))
            y1 = get_torch_empty_tensor(y1_pt.shape, dtype)
            y2 = get_torch_empty_tensor(y2_pt.shape, dtype)

            # Run AITemplate module.
            module.run_with_tensors([input_pt], [y1, y2])

            # Do comparisons.
            self.assertTrue(torch.allclose(y1, y1_pt, atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(y2, y2_pt, atol=1e-2, rtol=1e-2))

    @unittest.skipIf(
        detect_target().name() == "cuda" and int(detect_target()._arch) < 80,
        "Not supported by cuda sm<80",
    )
    def test_two_views(self):
        self._test_two_parallel_views()
        self._test_two_serial_view_outputs()

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_two_views_float(self):
        self._test_two_parallel_views(dtype="float")
        self._test_two_serial_view_outputs(dtype="float")


if __name__ == "__main__":
    unittest.main()
