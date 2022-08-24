# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import shape_utils


class TransformOddAlignmentCase(unittest.TestCase):
    def _create_permute_bmm_graph(self, A_shape, B_shape, bmm_type):
        OP = getattr(ops, bmm_type, None)
        assert OP is not None

        A = Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        B = Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)

        Y = OP()(A, B)
        Y._attrs["name"] = "target_bmm_tensor"
        return Y

    def _extract_src_op(self, tensors):
        ret = []
        for tensor in tensors:
            if len(tensor.src_ops()) != 1:
                ret.append(None)
            else:
                ret.append(list(tensor.src_ops())[0])

        return ret

    def test_permute_bmm_A(self):
        B = [1, 3]
        M = 8
        K = 7
        N = 16
        batch_dim = shape_utils.gen_int_var_min_max(B)
        shape_A = [batch_dim, M, K]
        shape_B = [batch_dim, K, N]

        bmm_tensor = self._create_permute_bmm_graph(shape_A, shape_B, "bmm_rrr")

        output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = gen_execution_module(
            output, target, "./tmp", "alignment_permute_bmm_A"
        )

        exist_new_bmm = False
        for tensor in module.debug_sorted_graph:
            src_ops = tensor.src_ops()
            if len(src_ops) == 0:
                continue
            self.assertEqual(
                len(src_ops),
                1,
                "constructed graph should only have single-source op tensors",
            )
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"].startswith("bmm"):
                self.assertEqual(src_op._attrs["op"], "bmm_crr")
                exist_new_bmm = True

                inputs_op = self._extract_src_op(src_op._attrs["inputs"])
                self.assertEqual(inputs_op[0]._attrs["op"], "permute021")
        self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

        for b in B:
            X_pt = torch.randn(b, M, K).cuda().half()
            W_pt = torch.randn(b, K, N).cuda().half()
            Y_pt = torch.cos(torch.bmm(X_pt, W_pt))

            y = torch.empty([b, M, N]).cuda().half()
            module.RunWithTensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_permute_bmm_B(self):
        B = [1, 3]
        M = 8
        K = 16
        N = 7
        batch_dim = shape_utils.gen_int_var_min_max(B)
        shape_A = [batch_dim, M, K]
        shape_B = [batch_dim, K, N]

        bmm_tensor = self._create_permute_bmm_graph(shape_A, shape_B, "bmm_rrr")

        output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = gen_execution_module(
            output, target, "./tmp", "alignment_permute_bmm_B"
        )

        exist_new_bmm = False
        for tensor in module.debug_sorted_graph:
            src_ops = tensor.src_ops()
            if len(src_ops) == 0:
                continue
            self.assertEqual(
                len(src_ops),
                1,
                "constructed graph should only have single-source op tensors",
            )
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"].startswith("bmm"):
                self.assertEqual(src_op._attrs["op"], "bmm_rcr")
                exist_new_bmm = True

                inputs_op = self._extract_src_op(src_op._attrs["inputs"])
                self.assertEqual(inputs_op[1]._attrs["op"], "permute021")
        self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

        for b in B:
            X_pt = torch.randn(b, M, K).cuda().half()
            W_pt = torch.randn(b, K, N).cuda().half()
            Y_pt = torch.cos(torch.bmm(X_pt, W_pt))

            y = torch.empty([b, M, N]).cuda().half()
            module.RunWithTensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_permute_bmm_both(self):
        B = [1, 3]
        M = 8
        K = 7
        N = 16
        batch_dim = shape_utils.gen_int_var_min_max(B)
        shape_A = [batch_dim, M, K]
        shape_B = [batch_dim, N, K]

        bmm_tensor = self._create_permute_bmm_graph(shape_A, shape_B, "bmm_rcr")

        output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = gen_execution_module(
            output, target, "./tmp", "alignment_permute_bmm_both"
        )

        exist_new_bmm = False
        for tensor in module.debug_sorted_graph:
            src_ops = tensor.src_ops()
            if len(src_ops) == 0:
                continue
            self.assertEqual(
                len(src_ops),
                1,
                "constructed graph should only have single-source op tensors",
            )
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"].startswith("bmm"):
                self.assertEqual(src_op._attrs["op"], "bmm_crr")
                exist_new_bmm = True

                inputs_op = self._extract_src_op(src_op._attrs["inputs"])
                self.assertEqual(inputs_op[0]._attrs["op"], "permute021")
                self.assertEqual(inputs_op[1]._attrs["op"], "permute021")
        self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

        for b in B:
            X_pt = torch.randn(b, M, K).cuda().half()
            W_pt = torch.randn(b, N, K).cuda().half()
            Y_pt = torch.cos(torch.bmm(X_pt, W_pt.transpose(2, 1)))

            y = torch.empty([b, M, N]).cuda().half()
            module.RunWithTensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_permute_bmm_epilogue(self):
        B = [1, 3]
        M = 8
        K = 7
        N = 16
        batch_dim = shape_utils.gen_int_var_min_max(B)
        shape_A = [batch_dim, M, K]
        shape_B = [batch_dim, K, N]
        shape_D = [batch_dim, M, N]

        D = Tensor(shape=shape_D, dtype="float16", name="input_2", is_input=True)

        bmm_tensor = self._create_permute_bmm_graph(shape_A, shape_B, "bmm_rrr")
        add_tensor = ops.elementwise(FuncEnum.ADD)(bmm_tensor, D)

        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = gen_execution_module(
            output, target, "./tmp", "alignment_permute_bmm_epilogue"
        )

        exist_new_bmm = False
        for tensor in module.debug_sorted_graph:
            src_ops = tensor.src_ops()
            if len(src_ops) == 0:
                continue
            self.assertEqual(
                len(src_ops),
                1,
                "constructed graph should only have single-source op tensors",
            )
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"].startswith("bmm"):
                self.assertEqual(src_op._attrs["op"], "bmm_crr_add")
                exist_new_bmm = True

                inputs_op = self._extract_src_op(src_op._attrs["inputs"])
                self.assertEqual(inputs_op[0]._attrs["op"], "permute021")
        self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

        for b in B:
            X_pt = torch.randn(b, M, K).cuda().half()
            W_pt = torch.randn(b, K, N).cuda().half()
            D_pt = torch.randn(b, M, N).cuda().half()
            Y_pt = torch.cos(torch.bmm(X_pt, W_pt) + D_pt)

            y = torch.empty([b, M, N]).cuda().half()
            module.RunWithTensors(
                {"input_0": X_pt, "input_1": W_pt, "input_2": D_pt}, [y]
            )
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
