# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest
from typing import List

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntVar
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils
from parameterized import param, parameterized

try:
    # When this test is run standalone, or through pytest.
    import test_strided_view_utils as utils
except ImportError:
    # When this test is run as a buck target.
    from . import test_strided_view_utils as utils


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_{param.args[0]}"


class StridedViewCatOpTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "gemm_reshape_cat_fusible_simple",
                n=2,
                new_shape=[-1, 2, 2],
                cat_dim=2,
                expected_num_tensors=9,
                expected_num_ops=8,
            ),
            param(
                "gemm_reshape_cat_fusible_expand_1",
                n=2,
                new_shape=[-1, 2, 1, 2],
                cat_dim=3,
                expected_num_tensors=9,
                expected_num_ops=8,
            ),
            param(
                "gemm_reshape_cat_fusible_expand_2",
                n=4,
                new_shape=[-1, 4, 4, 1],
                cat_dim=2,
                expected_num_tensors=9,
                expected_num_ops=8,
            ),
            param(
                "gemm_reshape_cat_fusible_expand_3",
                n=2,
                new_shape=[-1, 2, 2, 1],
                cat_dim=2,
                expected_num_tensors=9,
                expected_num_ops=8,
            ),
            param(
                "gemm_reshape_cat_fusible_expand_4",
                n=4,
                new_shape=[-1, 4, 2, 2],
                cat_dim=2,
                expected_num_tensors=9,
                expected_num_ops=8,
            ),
            param(
                "gemm_reshape_cat_non_fusible_dynamic_dim",
                n=2,
                new_shape=[-1, 2],
                cat_dim=1,
                expected_num_tensors=20,
                expected_num_ops=14,
            ),
            param(
                "gemm_reshape_cat_non_fusible_stride_dim",
                n=2,
                new_shape=[-1, 2 * 2],
                cat_dim=1,
                expected_num_tensors=14,
                expected_num_ops=8,
            ),
            param(
                "gemm_reshape_cat_non_fusible_expand",
                n=4,
                new_shape=[-1, 4, 2, 2],
                cat_dim=3,
                expected_num_tensors=14,
                expected_num_ops=8,
            ),
        ],
        name_func=custom_name_func,
    )
    def test_strided_gemm_view_cat_fusible(
        self,
        test_name: str,
        n: int,
        new_shape: List[int],
        cat_dim: int,
        expected_num_tensors: int,
        expected_num_ops: int,
    ):
        batch_dim = IntVar([1, 2, 3], "batch_size")
        input0 = utils.gen_input_tensor([batch_dim, n, n], name="input0")
        input1 = utils.gen_input_tensor([n, n], name="input1")
        input2 = utils.gen_input_tensor([batch_dim, n, n], name="input2")
        input3 = utils.gen_input_tensor([n], name="input3")
        input4 = utils.gen_input_tensor([batch_dim, n, n], name="input4")
        input5 = utils.gen_input_tensor([batch_dim, n, n], name="input5")

        X0 = ops.gemm_rcr()(input0, input1)
        X1 = ops.gemm_rcr_bias()(input0, input1, input3)
        X2 = ops.gemm_rcr_bias_add()(input0, input1, input3, input4)
        X3 = ops.gemm_rcr_bias_add_add()(input0, input1, input3, input4, input4)
        X4 = ops.bmm_rcr()(input0, input2)

        # For now these ops do not support output_accessors yet.
        # TODO: enable these checks once these ops support output_accessors.
        X5 = ops.bmm_rrr_add()(input0, input2, input3)

        Xs = [X2, X1, X0, X3, X4, X5]
        Ys = [ops.reshape()(X, new_shape) for X in Xs]
        Ys.insert(2, ops.reshape()(input5, new_shape))
        Z = ops.concatenate()(Ys, dim=cat_dim)
        Z._attrs["name"] = "output0"
        Z._attrs["is_output"] = True

        # Gen module.
        target = detect_target()
        module = gen_execution_module([Z], target, "./tmp", test_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), expected_num_tensors)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)

        # Prepae PyTorch tensors.
        for batch_size in batch_dim._attrs["values"]:
            input0_pt = torch.randn([batch_size, n, n]).cuda().half()
            input1_pt = torch.randn([n, n]).cuda().half()
            input2_pt = torch.randn([batch_size, n, n]).cuda().half()
            input3_pt = torch.randn([n]).cuda().half()
            input4_pt = torch.randn([batch_size, n, n]).cuda().half()
            input5_pt = torch.randn([batch_size, n, n]).cuda().half()

            # Run PyTorch baseline.
            x0_pt = torch.nn.functional.linear(input0_pt, input1_pt)
            x1_pt = torch.nn.functional.linear(input0_pt, input1_pt, input3_pt)
            x2_pt = (
                torch.nn.functional.linear(input0_pt, input1_pt, input3_pt) + input4_pt
            )
            x3_pt = (
                torch.nn.functional.linear(input0_pt, input1_pt, input3_pt)
                + input4_pt
                + input4_pt
            )
            x4_pt = torch.bmm(input0_pt, input2_pt.transpose(1, 2))
            x5_pt = torch.bmm(input0_pt, input2_pt) + input3_pt

            xs_pt = [x2_pt, x1_pt, x0_pt, x3_pt, x4_pt, x5_pt]
            ys_pt = [torch.reshape(x, new_shape) for x in xs_pt]
            ys_pt.insert(2, torch.reshape(input5_pt, new_shape))
            z_pt = torch.cat(ys_pt, dim=cat_dim)
            z = torch.empty(z_pt.shape).cuda().half()

            # Run AITemplate module.
            module.RunWithTensors(
                {
                    "input0": input0_pt,
                    "input1": input1_pt,
                    "input2": input2_pt,
                    "input3": input3_pt,
                    "input4": input4_pt,
                    "input5": input5_pt,
                },
                [z],
            )

            # Do comparisons.
            self.assertTrue(
                torch.allclose(z, z_pt, atol=1e-2, rtol=1e-2),
                f"batch_size: {batch_size}, z: {z}, z_pt: {z_pt}, input5_pt: {input5_pt}",
            )


if __name__ == "__main__":
    unittest.main()
