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

from typing import List, Optional

import torch

from aitemplate.compiler import compile_model, Model, ops
from aitemplate.compiler.base import IntVar
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, test_utils
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils

from parameterized import param, parameterized


def _gen_fusible_view_ops_before_strided_op(
    name: str, batch_dim: Optional[IntVar], n1: int, n2: int
) -> List[Tensor]:
    assert n2 % 2 == 0, f"n2 must be even! n2: {n2}"
    target = detect_target()
    support_float = target.name() == "cuda" and int(target._arch) >= 80
    if batch_dim is not None:
        test_ops = [
            ops.reshape()(
                test_utils.gen_input_tensor(
                    [batch_dim, n1 * n2], name=name, dtype="float16"
                ),
                [-1, n1, n2],
            ),
            ops.flatten(start_dim=2, end_dim=-1)(
                test_utils.gen_input_tensor(
                    [batch_dim, n1, int(n2 / 2), 2], name=name, dtype="float16"
                )
            ),
            ops.squeeze(dim=1)(
                test_utils.gen_input_tensor(
                    [batch_dim, 1, n1, n2], name=name, dtype="float16"
                )
            ),
        ]
        if support_float:
            test_ops.append(
                ops.reshape()(
                    test_utils.gen_input_tensor(
                        [batch_dim, n1 * n2], name=name, dtype="float"
                    ),
                    [-1, n1, n2],
                )
            )
    else:
        test_ops = [
            ops.reshape()(
                test_utils.gen_input_tensor([n1 * n2], name=name, dtype="float16"),
                [n1, n2],
            ),
            ops.flatten(start_dim=1, end_dim=-1)(
                test_utils.gen_input_tensor(
                    [n1, int(n2 / 2), 2], name=name, dtype="float16"
                )
            ),
            ops.squeeze(dim=0)(
                test_utils.gen_input_tensor([1, n1, n2], name=name, dtype="float16")
            ),
        ]
        if support_float:
            test_ops.append(
                ops.flatten(start_dim=1, end_dim=-1)(
                    test_utils.gen_input_tensor(
                        [n1, int(n2 / 2), 2], name=name, dtype="float"
                    )
                ),
            )
    return test_ops


def _gen_non_fusible_view_ops_before_strided_op(
    name: str, batch_dim: IntVar, n1: int, n2: int
) -> List[Tensor]:
    new_batch_dim = IntVar(
        name=batch_dim._attrs["name"],
        values=[int(value / 2) for value in batch_dim._attrs["values"]],
    )
    test_ops = [
        ops.reshape()(
            test_utils.gen_input_tensor(
                [new_batch_dim, n1, n2 * 2], name=name, dtype="float16"
            ),
            [-1, n1, n2],
        ),
        ops.flatten(start_dim=0, end_dim=1)(
            test_utils.gen_input_tensor(
                [new_batch_dim, 2, n1, n2], name=name, dtype="float16"
            )
        ),
    ]
    target = detect_target()
    if target.name() == "cuda" and int(target._arch) >= 80:
        test_ops.append(
            ops.reshape()(
                test_utils.gen_input_tensor(
                    [new_batch_dim, n1, n2 * 2], name=name, dtype="float"
                ),
                [-1, n1, n2],
            )
        )
    return test_ops


def _gen_multiple_fusible_view_ops_before_strided_op(
    name: str, batch_dim: IntVar, n1: int, n2: int
) -> List[Tensor]:
    test_ops = [
        ops.reshape()(
            ops.reshape()(
                test_utils.gen_input_tensor(
                    [batch_dim, n1, n2], name=name, dtype="float16"
                ),
                [-1, n1 * n2],
            ),
            [-1, n1, n2],
        ),
        ops.squeeze(dim=1)(
            ops.unsqueeze(dim=1)(
                test_utils.gen_input_tensor(
                    [batch_dim, n1, n2], name=name, dtype="float16"
                )
            ),
        ),
    ]
    target = detect_target()
    if target.name() == "cuda" and int(target._arch) >= 80:
        test_ops.append(
            ops.squeeze(dim=1)(
                ops.unsqueeze(dim=1)(
                    test_utils.gen_input_tensor(
                        [batch_dim, n1, n2], name=name, dtype="float16"
                    )
                )
            )
        )
    return test_ops


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_{param.args[0]}"


class ViewStridedOpTestCase(unittest.TestCase):
    def _gen_view_bmm_module(
        self,
        input0: Tensor,
        input1: Tensor,
        test_name: str,
        dtype: str,
        expected_num_tensors: int,
        expected_num_ops: int,
        num_bmms: int = 1,
    ) -> Model:
        Ys = []
        for i in range(num_bmms):
            Y = ops.bmm_rcr()(input0, input1)
            Y._attrs["name"] = f"output{str(i)}"
            Y._attrs["is_output"] = True
            Ys.append(Y)

        # Gen module.
        target = detect_target()
        module = compile_model(Ys, target, "./tmp", f"{test_name}_{dtype}")

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), expected_num_tensors)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)

        return module

    def _test_view_and_bmm(
        self,
        module: Model,
        x0_pt: torch.Tensor,
        x1_pt: torch.Tensor,
        ys: List[torch.Tensor],
        x0_shape: List[int],
        x1_shape: List[int],
    ):
        # Run PyTorch baseline.
        y_pts = []
        for _ in range(len(ys)):
            y_pt = torch.matmul(x0_pt, x1_pt.transpose(1, 2))
            y_pts.append(y_pt)

        # Run AITemplate module.
        inputs = [x0_pt.reshape(*x0_shape), x1_pt.reshape(*x1_shape)]
        module.run_with_tensors(inputs, ys)

        # Do comparisons.
        for y, y_pt in zip(ys, y_pts):
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        [
            param(
                f"single_{test_utils.get_src_op_name(tensor0)}_"
                f"{test_utils.get_src_op_name(tensor0)}_bmm_fusion",
                tensor0,
                tensor1,
                tensor0._attrs["dtype"],
            )
            for tensor0, tensor1 in zip(
                _gen_fusible_view_ops_before_strided_op(
                    "input0",
                    batch_dim=IntVar([1, 128, 256], "batch_size"),
                    n1=13,
                    n2=46,
                ),
                _gen_fusible_view_ops_before_strided_op(
                    "input1", batch_dim=IntImm(1), n1=5, n2=46
                ),
            )
        ],
        name_func=custom_name_func,
    )
    def test_single_view_and_bmm_fusible(
        self, test_name: str, input0: Tensor, input1: Tensor, dtype: str
    ):
        orig_a_shape = test_utils.get_src_input(input0)._attrs["shape"]
        orig_b_shape = test_utils.get_src_input(input1)._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0, input1, test_name, dtype, expected_num_tensors=3, expected_num_ops=1
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = get_random_torch_tensor(
                [batch_size, a_shape[1].value(), a_shape[2].value()], dtype
            )
            x1_pt = get_random_torch_tensor([dim.value() for dim in b_shape], dtype)
            y = get_torch_empty_tensor(
                [batch_size, a_shape[1].value(), b_shape[1].value()], dtype
            )
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y],
                test_utils.get_shape(orig_a_shape, dim_to_value_dict),
                test_utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    @parameterized.expand(
        [
            param(
                f"single_{test_utils.get_src_op_name(tensor0)}_"
                f"{test_utils.get_src_op_name(tensor0)}_multi_bmm_fusion",
                tensor0,
                tensor1,
                tensor0._attrs["dtype"],
            )
            for tensor0, tensor1 in zip(
                _gen_fusible_view_ops_before_strided_op(
                    "input0",
                    batch_dim=IntVar([1, 128, 256], "batch_size"),
                    n1=13,
                    n2=46,
                ),
                _gen_fusible_view_ops_before_strided_op(
                    "input1", batch_dim=IntImm(1), n1=5, n2=46
                ),
            )
        ],
        name_func=custom_name_func,
    )
    def test_single_view_and_multi_bmm_fusible(
        self, test_name: str, input0: Tensor, input1: Tensor, dtype: str
    ):
        orig_a_shape = test_utils.get_src_input(input0)._attrs["shape"]
        orig_b_shape = test_utils.get_src_input(input1)._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0,
            input1,
            test_name,
            dtype,
            expected_num_tensors=4,
            expected_num_ops=2,
            num_bmms=2,
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = get_random_torch_tensor(
                [batch_size, a_shape[1].value(), a_shape[2].value()], dtype
            )
            x1_pt = get_random_torch_tensor([dim.value() for dim in b_shape], dtype)
            y0 = get_torch_empty_tensor(
                [batch_size, a_shape[1].value(), b_shape[1].value()], dtype
            )
            y1 = y0.clone()
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y0, y1],
                test_utils.get_shape(orig_a_shape, dim_to_value_dict),
                test_utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    def _test_multi_view_and_multi_bmm_fusible(self, dtype="float16"):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N0 = 13
        N1 = 46
        N2 = 5
        X0 = test_utils.gen_input_tensor(
            [batch_dim, N0 * N1], name="input0", dtype=dtype
        )
        X1 = test_utils.gen_input_tensor([1, N2 * N1], name="input1", dtype=dtype)
        X2 = ops.reshape()(X0, [-1, N0, N1])
        X3 = ops.reshape()(X0, [-1, N0, N1])
        X4 = ops.reshape()(X1, [-1, N2, N1])
        X5 = ops.reshape()(X1, [-1, N2, N1])

        orig_a_shape = X0._attrs["shape"]
        orig_b_shape = X1._attrs["shape"]

        Ys = []
        Y0 = ops.bmm_rcr()(X2, X4)
        Y1 = ops.bmm_rcr()(X3, X5)
        Ys = [Y0, Y1]
        for i, Y in enumerate(Ys):
            Y._attrs["name"] = f"output{str(i)}"
            Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(
            Ys, target, "./tmp", f"multi_view_multi_bmm_fusion_{dtype}"
        )

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        a_shape = X2._attrs["shape"]
        b_shape = X4._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = get_random_torch_tensor(
                [batch_size, a_shape[1].value(), a_shape[2].value()], dtype
            )
            x1_pt = get_random_torch_tensor([dim.value() for dim in b_shape], dtype)
            y0 = get_torch_empty_tensor(
                [batch_size, a_shape[1].value(), b_shape[1].value()], dtype
            )
            y1 = y0.clone()
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y0, y1],
                test_utils.get_shape(orig_a_shape, dim_to_value_dict),
                test_utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    def test_multi_view_and_multi_bmm_fusible(self):
        self._test_multi_view_and_multi_bmm_fusible()

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_multi_view_and_multi_bmm_fusible_fp32_sm80(self):
        self._test_multi_view_and_multi_bmm_fusible(dtype="float")

    @parameterized.expand(
        [
            param(
                f"multi_{test_utils.get_src_op_name(tensor0)}_"
                f"{test_utils.get_src_op_name(tensor0)}_bmm_fusion",
                tensor0,
                tensor1,
                tensor0._attrs["dtype"],
            )
            for tensor0, tensor1 in zip(
                _gen_multiple_fusible_view_ops_before_strided_op(
                    "input0",
                    batch_dim=IntVar([1, 128, 256], "batch_size"),
                    n1=13,
                    n2=46,
                ),
                _gen_multiple_fusible_view_ops_before_strided_op(
                    "input1", batch_dim=IntImm(1), n1=5, n2=46
                ),
            )
        ],
        name_func=custom_name_func,
    )
    def test_multiple_view_and_bmm_fusible(
        self, test_name: str, input0: Tensor, input1: Tensor, dtype: str
    ):
        orig_a_shape = test_utils.get_src_input(
            test_utils.get_src_input(input0)
        )._attrs["shape"]
        orig_b_shape = test_utils.get_src_input(
            test_utils.get_src_input(input1)
        )._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0, input1, test_name, dtype, expected_num_tensors=3, expected_num_ops=1
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = get_random_torch_tensor(
                [batch_size, a_shape[1].value(), a_shape[2].value()], dtype
            )
            x1_pt = get_random_torch_tensor([dim.value() for dim in b_shape], dtype)
            y = get_torch_empty_tensor(
                [batch_size, a_shape[1].value(), b_shape[1].value()], dtype
            )
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y],
                test_utils.get_shape(orig_a_shape, dim_to_value_dict),
                test_utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    @parameterized.expand(
        [
            param(
                f"non_fusible_{test_utils.get_src_op_name(tensor0)}_"
                f"{test_utils.get_src_op_name(tensor0)}_bmm_fusion",
                tensor0,
                tensor1,
                tensor0._attrs["dtype"],
            )
            for tensor0, tensor1 in zip(
                _gen_non_fusible_view_ops_before_strided_op(
                    "input0",
                    batch_dim=IntVar([2, 128, 256], "batch_size"),
                    n1=13,
                    n2=46,
                ),
                _gen_non_fusible_view_ops_before_strided_op(
                    "input1",
                    batch_dim=IntVar([2, 128, 256], "batch_size"),
                    n1=5,
                    n2=46,
                ),
            )
        ],
        name_func=custom_name_func,
    )
    def test_non_fusible_view_and_bmm(
        self, test_name: str, input0: Tensor, input1: Tensor, dtype: str
    ):
        orig_a_shape = test_utils.get_src_input(input0)._attrs["shape"]
        orig_b_shape = test_utils.get_src_input(input1)._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0, input1, test_name, dtype, expected_num_tensors=5, expected_num_ops=3
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = get_random_torch_tensor(
                [batch_size, a_shape[1].value(), a_shape[2].value()], dtype
            )
            x1_pt = get_random_torch_tensor(
                [batch_size, b_shape[1].value(), b_shape[2].value()], dtype
            )
            y = get_torch_empty_tensor(
                [batch_size, a_shape[1].value(), b_shape[1].value()], dtype
            )
            dim_to_value_dict = {"batch_size": int(batch_size / 2)}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y],
                test_utils.get_shape(orig_a_shape, dim_to_value_dict),
                test_utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    def _test_single_view_and_gemm_fusible(self, dtype="float16"):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N0 = 13
        N1 = 46
        N2 = 6
        X0 = test_utils.gen_input_tensor(
            [batch_dim, N0 * N1], name="input0", dtype=dtype
        )
        X1 = test_utils.gen_input_tensor([1, N2 * N1], name="input1", dtype=dtype)
        X2 = test_utils.gen_input_tensor([N2], name="input2", dtype=dtype)
        X3 = ops.reshape()(X0, [-1, N0, N1])
        X4 = ops.reshape()(X1, [N2, N1])
        X5 = ops.reshape()(X1, [N1, N2])

        Ys = []
        Y0 = ops.gemm_rcr()(X3, X4)
        Y1 = ops.gemm_rcr_bias()(X3, X4, X2)
        Y2 = ops.gemm_rrr()(X3, X5)
        Ys = [Y0, Y1, Y2]
        for i, Y in enumerate(Ys):
            Y._attrs["name"] = f"output{str(i)}"
            Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = compile_model(Ys, target, "./tmp", f"single_view_gemm_fusion_{dtype}")

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 6)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            x0_pt = get_random_torch_tensor([batch_size, N0 * N1], dtype)
            x1_pt = get_random_torch_tensor([1, N2 * N1], dtype)
            x2_pt = get_random_torch_tensor([N2], dtype)
            x3_pt = torch.reshape(x0_pt, [-1, N0, N1])
            x4_pt = torch.reshape(x1_pt, [N2, N1])
            x5_pt = torch.reshape(x1_pt, [N1, N2])
            y0_pt = torch.nn.functional.linear(x3_pt, x4_pt)
            y1_pt = torch.nn.functional.linear(x3_pt, x4_pt) + x2_pt
            y2_pt = torch.nn.functional.linear(x3_pt, x5_pt.transpose(0, 1))
            y_pts = [y0_pt, y1_pt, y2_pt]
            ys = [
                get_torch_empty_tensor([batch_size, N0, N2], dtype),
                get_torch_empty_tensor([batch_size, N0, N2], dtype),
                get_torch_empty_tensor([batch_size, N0, N2], dtype),
            ]

            # Run AITemplate module.
            inputs = [x0_pt, x1_pt, x2_pt]
            module.run_with_tensors(inputs, ys)

            # Do comparisons.
            for y, y_pt in zip(ys, y_pts):
                self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    def test_single_view_and_gemm_fusible(self):
        self._test_single_view_and_gemm_fusible()

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_single_view_and_gemm_fusible_fp32_sm80(self):
        self._test_single_view_and_gemm_fusible(dtype="float")


filter_test_cases_by_test_env(ViewStridedOpTestCase)

if __name__ == "__main__":
    unittest.main()
