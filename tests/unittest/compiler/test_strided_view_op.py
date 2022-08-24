# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest
from functools import partial

from typing import Callable, Dict, List, Tuple

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils

from parameterized import param, parameterized

try:
    # When this test is run standalone, or through pytest.
    import test_strided_view_utils as utils
except ImportError:
    # When this test is run as a buck target.
    from . import test_strided_view_utils as utils


def _gen_simple_strided_ops(
    batch_dim: IntVar, n1: int, n2: int
) -> List[Tuple[Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
    return [
        (
            "tanh",
            ops.elementwise(FuncEnum.TANH)(utils.gen_input_tensor([batch_dim, n1, n2])),
            torch.tanh,
        ),
        (
            "layernorm",
            ops.layernorm(normalized_shape=[IntImm(n2)])(
                utils.gen_input_tensor([batch_dim, n1, n2])
            ),
            partial(torch.nn.functional.layer_norm, normalized_shape=[n2]),
        ),
        (
            "sum",
            ops.reduce_sum(2, keepdim=True)(
                utils.gen_input_tensor([batch_dim, n1, n2])
            ),
            partial(torch.sum, dim=2, keepdim=True),
        ),
    ]


def _gen_fusible_view_ops_after_strided_op() -> Dict[str, Callable[[Tensor], Tensor]]:
    def reshape_op(input_tensor: Tensor):
        shape = input_tensor._attrs["shape"]
        return ops.reshape()(
            input_tensor,
            [-1, shape[1].value() * shape[2].value()],
        )

    def flatten_op(input_tensor: Tensor):
        return ops.flatten(start_dim=1, end_dim=-1)(input_tensor)

    return {"reshape": reshape_op, "flatten": flatten_op}


def _gen_non_fusible_view_ops_after_strided_op() -> Dict[
    str, Callable[[Tensor], Tensor]
]:
    def reshape_op(input_tensor: Tensor):
        n2 = input_tensor._attrs["shape"][2].value()
        return ops.reshape()(input_tensor, [-1, n2])

    def flatten_op(input_tensor: Tensor):
        return ops.flatten(start_dim=0, end_dim=1)(input_tensor)

    return {"reshape": reshape_op, "flatten": flatten_op}


def _gen_multiple_fusible_view_ops_after_strided_op() -> Dict[
    str, Callable[[Tensor], Tensor]
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

    return {
        "multi_reshape": multi_reshape,
        "squeeze_unsqueeze": squeeze_unsqueeze,
    }


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_{param.args[0]}"


class StridedViewOpTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            param(f"single_gemm_{name}_fusion", func)
            for (name, func) in _gen_fusible_view_ops_after_strided_op().items()
        ],
        name_func=custom_name_func,
    )
    def test_single_gemm_and_view_fusible(self, test_name, func):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        K = 10
        input0 = utils.gen_input_tensor([batch_dim, N1, K])
        input1 = utils.gen_input_tensor([N2, K])
        X0 = ops.gemm_rcr()(input0, input1)
        Y = ops.elementwise(FuncEnum.TANH)(func(X0))
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module([Y], target, "./tmp", test_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input0_pt = torch.randn(batch_size, N1, K).cuda().half()
            input1_pt = torch.randn(N2, K).cuda().half()

            # Run PyTorch baseline.
            x0_pt = torch.matmul(input0_pt, input1_pt.transpose(0, 1))
            dim_to_value_dict = {"batch_size": batch_size}
            y_pt = torch.tanh(
                torch.reshape(
                    x0_pt, utils.get_shape(Y._attrs["shape"], dim_to_value_dict)
                )
            )
            y = torch.empty(y_pt.shape).cuda().half()

            # Run AITemplate module.
            module.RunWithTensors([input0_pt, input1_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        [
            param(f"single_bmm_{name}_fusion", func)
            for (
                name,
                func,
            ) in _gen_multiple_fusible_view_ops_after_strided_op().items()
        ],
        name_func=custom_name_func,
    )
    def test_single_bmm_and_multi_view_fusible(self, test_name, func):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        K = 10
        input0 = utils.gen_input_tensor([batch_dim, N1, K])
        input1 = utils.gen_input_tensor([batch_dim, K, N2])
        X0 = ops.bmm_rrr()(input0, input1)
        Y = ops.elementwise(FuncEnum.COS)(func(X0))
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module([Y], target, "./tmp", test_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input0_pt = torch.randn(batch_size, N1, K).cuda().half()
            input1_pt = torch.randn(batch_size, K, N2).cuda().half()

            # Run PyTorch baseline.
            x0_pt = torch.matmul(input0_pt, input1_pt)
            dim_to_value_dict = {"batch_size": batch_size}
            y_pt = torch.cos(
                torch.reshape(
                    x0_pt, utils.get_shape(Y._attrs["shape"], dim_to_value_dict)
                )
            )
            y = torch.empty(y_pt.shape).cuda().half()

            # Run AITemplate module.
            module.RunWithTensors([input0_pt, input1_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        [
            param(f"single_{op_name}_reshape_fusion", input_tensor, torch_func)
            for (op_name, input_tensor, torch_func) in _gen_simple_strided_ops(
                IntVar([1, 128, 256], "batch_size"), n1=10, n2=8
            )
        ],
        name_func=custom_name_func,
    )
    def test_single_op_and_view_fusible(self, test_name, input_tensor, torch_func):
        src_input = utils.get_src_input(input_tensor)
        batch_dim = src_input._attrs["shape"][0]
        n1 = src_input._attrs["shape"][1].value()
        n2 = src_input._attrs["shape"][2].value()
        Y = ops.elementwise(FuncEnum.TANH)(ops.reshape()(input_tensor, [batch_dim, -1]))
        Y._attrs["name"] = "output0"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module([Y], target, "./tmp", test_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input_pt = torch.randn(batch_size, n1, n2).cuda().half()

            # Run PyTorch baseline.
            y_pt = torch.tanh(torch.reshape(torch_func(input_pt), [batch_size, -1]))
            y = torch.empty(y_pt.shape).cuda().half()

            # Run AITemplate module.
            module.RunWithTensors([input_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        [
            param(f"single_op_{name}_non_fusion", func)
            for (name, func) in _gen_non_fusible_view_ops_after_strided_op().items()
        ],
        name_func=custom_name_func,
    )
    def test_single_op_and_view_non_fusible(self, test_name, func):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        X0 = utils.gen_input_tensor([batch_dim, N1, N2])
        X1 = ops.elementwise(FuncEnum.TANH)(X0)
        Y = ops.elementwise(FuncEnum.TANH)(func(X1))
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module([Y], target, "./tmp", test_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 4)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 3)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            x0_pt = torch.randn(batch_size, N1, N2).cuda().half()

            # Run PyTorch baseline.
            y_pt = torch.tanh(torch.reshape(torch.tanh(x0_pt), [-1, N2]))
            y = torch.empty(y_pt.shape).cuda().half()

            # Run AITemplate module.
            module.RunWithTensors([x0_pt], [y])

            # Do comparisons.
            self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

    def test_two_serial_view_outputs(self):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        X0 = utils.gen_input_tensor([batch_dim, N1, N2])
        X1 = ops.elementwise(FuncEnum.TANH)(X0)
        Y1 = ops.reshape()(X1, [-1, N1 * N2])
        Y2 = ops.reshape()(Y1, [-1, N1, N2])
        Y1._attrs["name"] = "output1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "output2"
        Y2._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module([Y1, Y2], target, "./tmp", "two_view_outputs")

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 3)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 2)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input_pt = torch.randn(batch_size, N1, N2).cuda().half()

            # Run PyTorch baseline.
            y1_pt = torch.reshape(torch.tanh(input_pt), [batch_size, N1 * N2])
            y2_pt = torch.reshape(y1_pt, [batch_size, N1, N2])
            y1 = torch.empty(y1_pt.shape).cuda().half()
            y2 = torch.empty(y2_pt.shape).cuda().half()

            # Run AITemplate module.
            module.RunWithTensors([input_pt], [y1, y2])

            # Do comparisons.
            self.assertTrue(torch.allclose(y1, y1_pt, atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(y2, y2_pt, atol=1e-2, rtol=1e-2))

    def test_two_parallel_views(self):
        batch_dim = IntVar([1, 128, 256], "batch_size")
        N1 = 8
        N2 = 6
        X0 = utils.gen_input_tensor([batch_dim, N1, N2])
        X1 = ops.elementwise(FuncEnum.TANH)(X0)
        Y1 = ops.elementwise(FuncEnum.TANH)(ops.reshape()(X1, [-1, N1 * N2]))
        Y2 = ops.elementwise(FuncEnum.TANH)(ops.reshape()(X1, [-1, N1, N2]))
        Y1._attrs["name"] = "output1"
        Y1._attrs["is_output"] = True
        Y2._attrs["name"] = "output2"
        Y2._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module(
            [Y1, Y2], target, "./tmp", "two_parallel_view_outputs"
        )

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), 6)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), 5)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input_pt = torch.randn(batch_size, N1, N2).cuda().half()
            x1_pt = torch.tanh(input_pt)

            # Run PyTorch baseline.
            y1_pt = torch.tanh(torch.reshape(x1_pt, [batch_size, N1 * N2]))
            y2_pt = torch.tanh(torch.reshape(x1_pt, [batch_size, N1, N2]))
            y1 = torch.empty(y1_pt.shape).cuda().half()
            y2 = torch.empty(y2_pt.shape).cuda().half()

            # Run AITemplate module.
            module.RunWithTensors([input_pt], [y1, y2])

            # Do comparisons.
            self.assertTrue(torch.allclose(y1, y1_pt, atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(y2, y2_pt, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
