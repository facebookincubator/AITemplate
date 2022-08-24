# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

from typing import List, Optional

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntVar
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.testing.model import Model
from aitemplate.utils import graph_utils

from parameterized import param, parameterized

try:
    # When this test is run standalone, or through pytest.
    import test_strided_view_utils as utils
except ImportError:
    # When this test is run as a buck target.
    from . import test_strided_view_utils as utils


def _gen_fusible_view_ops_before_strided_op(
    name: str, batch_dim: Optional[IntVar], n1: int, n2: int
) -> List[Tensor]:
    assert n2 % 2 == 0, f"n2 must be even! n2: {n2}"
    if batch_dim is not None:
        return [
            ops.reshape()(
                utils.gen_input_tensor([batch_dim, n1 * n2], name),
                [-1, n1, n2],
            ),
            ops.flatten(start_dim=2, end_dim=-1)(
                utils.gen_input_tensor([batch_dim, n1, int(n2 / 2), 2], name)
            ),
            ops.squeeze(dim=1)(utils.gen_input_tensor([batch_dim, 1, n1, n2], name)),
        ]
    else:
        return [
            ops.reshape()(
                utils.gen_input_tensor([n1 * n2], name),
                [n1, n2],
            ),
            ops.flatten(start_dim=1, end_dim=-1)(
                utils.gen_input_tensor([n1, int(n2 / 2), 2], name)
            ),
            ops.squeeze(dim=0)(utils.gen_input_tensor([1, n1, n2], name)),
        ]


def _gen_non_fusible_view_ops_before_strided_op(
    name: str, batch_dim: IntVar, n1: int, n2: int
) -> List[Tensor]:
    new_batch_dim = IntVar(
        name=batch_dim._attrs["name"],
        values=[int(value / 2) for value in batch_dim._attrs["values"]],
    )
    return [
        ops.reshape()(
            utils.gen_input_tensor([new_batch_dim, n1, n2 * 2], name),
            [-1, n1, n2],
        ),
        ops.flatten(start_dim=0, end_dim=1)(
            utils.gen_input_tensor([new_batch_dim, 2, n1, n2], name)
        ),
    ]


def _gen_multiple_fusible_view_ops_before_strided_op(
    name: str, batch_dim: IntVar, n1: int, n2: int
) -> List[Tensor]:
    return [
        ops.reshape()(
            ops.reshape()(
                utils.gen_input_tensor([batch_dim, n1, n2], name),
                [-1, n1 * n2],
            ),
            [-1, n1, n2],
        ),
        ops.squeeze(dim=1)(
            ops.unsqueeze(dim=1)(utils.gen_input_tensor([batch_dim, n1, n2], name)),
        ),
    ]


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_{param.args[0]}"


class ViewStridedOpTestCase(unittest.TestCase):
    def _gen_view_bmm_module(
        self,
        input0: Tensor,
        input1: Tensor,
        test_name: str,
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
        module = gen_execution_module(Ys, target, "./tmp", test_name)

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
        module.RunWithTensors(inputs, ys)

        # Do comparisons.
        for y, y_pt in zip(ys, y_pts):
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        [
            param(
                f"single_{utils.get_src_op_name(tensor0)}_{utils.get_src_op_name(tensor0)}_bmm_fusion",
                tensor0,
                tensor1,
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
        self, test_name: str, input0: Tensor, input1: Tensor
    ):
        orig_a_shape = utils.get_src_input(input0)._attrs["shape"]
        orig_b_shape = utils.get_src_input(input1)._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0, input1, test_name, expected_num_tensors=3, expected_num_ops=1
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = (
                torch.randn(batch_size, a_shape[1].value(), a_shape[2].value())
                .cuda()
                .half()
            )
            x1_pt = torch.randn([dim.value() for dim in b_shape]).cuda().half()
            y = (
                torch.empty([batch_size, a_shape[1].value(), b_shape[1].value()])
                .cuda()
                .half()
            )
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y],
                utils.get_shape(orig_a_shape, dim_to_value_dict),
                utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    @parameterized.expand(
        [
            param(
                f"single_{utils.get_src_op_name(tensor0)}_{utils.get_src_op_name(tensor0)}_multi_bmm_fusion",
                tensor0,
                tensor1,
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
        self, test_name: str, input0: Tensor, input1: Tensor
    ):
        orig_a_shape = utils.get_src_input(input0)._attrs["shape"]
        orig_b_shape = utils.get_src_input(input1)._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0,
            input1,
            test_name,
            expected_num_tensors=4,
            expected_num_ops=2,
            num_bmms=2,
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = (
                torch.randn(batch_size, a_shape[1].value(), a_shape[2].value())
                .cuda()
                .half()
            )
            x1_pt = torch.randn([dim.value() for dim in b_shape]).cuda().half()
            y0 = (
                torch.empty([batch_size, a_shape[1].value(), b_shape[1].value()])
                .cuda()
                .half()
            )
            y1 = y0.clone()
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y0, y1],
                utils.get_shape(orig_a_shape, dim_to_value_dict),
                utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    def test_multi_view_and_multi_bmm_fusible(self):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N0 = 13
        N1 = 46
        N2 = 5
        X0 = utils.gen_input_tensor([batch_dim, N0 * N1], "input0")
        X1 = utils.gen_input_tensor([1, N2 * N1], "input1")
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
        module = gen_execution_module(
            Ys, target, "./tmp", "multi_view_multi_bmm_fusion"
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
            x0_pt = (
                torch.randn(batch_size, a_shape[1].value(), a_shape[2].value())
                .cuda()
                .half()
            )
            x1_pt = torch.randn([dim.value() for dim in b_shape]).cuda().half()
            y0 = (
                torch.empty([batch_size, a_shape[1].value(), b_shape[1].value()])
                .cuda()
                .half()
            )
            y1 = y0.clone()
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y0, y1],
                utils.get_shape(orig_a_shape, dim_to_value_dict),
                utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    @parameterized.expand(
        [
            param(
                f"multi_{utils.get_src_op_name(tensor0)}_{utils.get_src_op_name(tensor0)}_bmm_fusion",
                tensor0,
                tensor1,
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
        self, test_name: str, input0: Tensor, input1: Tensor
    ):
        orig_a_shape = utils.get_src_input(utils.get_src_input(input0))._attrs["shape"]
        orig_b_shape = utils.get_src_input(utils.get_src_input(input1))._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0, input1, test_name, expected_num_tensors=3, expected_num_ops=1
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = (
                torch.randn(batch_size, a_shape[1].value(), a_shape[2].value())
                .cuda()
                .half()
            )
            x1_pt = torch.randn([dim.value() for dim in b_shape]).cuda().half()
            y = (
                torch.empty([batch_size, a_shape[1].value(), b_shape[1].value()])
                .cuda()
                .half()
            )
            dim_to_value_dict = {"batch_size": batch_size}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y],
                utils.get_shape(orig_a_shape, dim_to_value_dict),
                utils.get_shape(orig_b_shape, dim_to_value_dict),
            )

    @parameterized.expand(
        [
            param(
                f"non_fusible_{utils.get_src_op_name(tensor0)}_{utils.get_src_op_name(tensor0)}_bmm_fusion",
                tensor0,
                tensor1,
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
        self, test_name: str, input0: Tensor, input1: Tensor
    ):
        orig_a_shape = utils.get_src_input(input0)._attrs["shape"]
        orig_b_shape = utils.get_src_input(input1)._attrs["shape"]

        # Gen module.
        module = self._gen_view_bmm_module(
            input0, input1, test_name, expected_num_tensors=5, expected_num_ops=3
        )

        # Prepae PyTorch tensors.
        a_shape = input0._attrs["shape"]
        b_shape = input1._attrs["shape"]
        for batch_size in a_shape[0]._attrs["values"]:
            x0_pt = (
                torch.randn(batch_size, a_shape[1].value(), a_shape[2].value())
                .cuda()
                .half()
            )
            x1_pt = (
                torch.randn(batch_size, b_shape[1].value(), b_shape[2].value())
                .cuda()
                .half()
            )
            y = (
                torch.empty([batch_size, a_shape[1].value(), b_shape[1].value()])
                .cuda()
                .half()
            )
            dim_to_value_dict = {"batch_size": int(batch_size / 2)}
            self._test_view_and_bmm(
                module,
                x0_pt,
                x1_pt,
                [y],
                utils.get_shape(orig_a_shape, dim_to_value_dict),
                utils.get_shape(orig_b_shape, dim_to_value_dict),
            )


if __name__ == "__main__":
    unittest.main()
