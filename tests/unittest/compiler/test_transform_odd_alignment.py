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

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import _TorchConstantTensorData, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils


def _extract_shape(batch, shape):
    if len(shape) == 2:
        return shape

    return (batch, shape[-2], shape[-1])


class TransformOddAlignmentCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def _create_permute_bmm_graph(
        self, A_shape, B_shape, bmm_type, const_A=None, const_B=None
    ):
        OP = getattr(ops, bmm_type, None)
        assert OP is not None

        A = (
            const_A
            if const_A
            else Tensor(shape=A_shape, dtype="float16", name="input_0", is_input=True)
        )
        B = (
            const_B
            if const_B
            else Tensor(shape=B_shape, dtype="float16", name="input_1", is_input=True)
        )

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

    def _test_permute_bmm_A(
        self,
        B,
        shape_A,
        shape_B,
        origin_bmm,
        target_bmm,
        is_const,
        is_elementwise=False,
        strided_output=True,
        test_prefix="",
    ):
        M = shape_A[-2] if origin_bmm[-3] == "r" else shape_A[-1]
        N = shape_B[-1] if origin_bmm[-2] == "r" else shape_B[-2]

        for b in B:
            const_A, const_B = None, None
            if is_elementwise:
                const_A = Tensor(
                    shape=shape_A, dtype="float16", name="input_0", is_input=True
                )
                const_A = ops.elementwise(FuncEnum.ADD)(const_A, const_A)
            elif is_const:
                const_A_data = torch.randn(_extract_shape(1, shape_A)).half().cuda()
                const_A = Tensor(
                    shape=_extract_shape(1, shape_A),
                    name="input_0",
                )
                const_A._bind_data(_TorchConstantTensorData(const_A_data))
            bmm_tensor = self._create_permute_bmm_graph(
                shape_A, shape_B, origin_bmm, const_A=const_A, const_B=const_B
            )

            if strided_output:
                output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
            else:
                output = bmm_tensor
            output._attrs["name"] = "output_0"
            output._attrs["is_output"] = True

            # Check value correctness
            target = detect_target()
            module = compile_model(
                output,
                target,
                "./tmp",
                f"{test_prefix}alignment_permute_bmm_A_{b}_{origin_bmm}_to_{target_bmm}_{is_const}",
            )

            exist_new_bmm = False
            for tensor in module.debug_sorted_graph:
                src_ops = tensor.src_ops()
                if len(src_ops) == 0:
                    continue
                if not is_elementwise:
                    self.assertEqual(
                        len(src_ops),
                        1,
                        "constructed graph should only have single-source op tensors",
                    )
                src_op = list(tensor.src_ops())[0]
                if src_op._attrs["op"].startswith("bmm"):
                    if not is_elementwise:
                        self.assertEqual(src_op._attrs["op"], target_bmm)
                    exist_new_bmm = True

                    if is_const:
                        continue
                    inputs_op = self._extract_src_op(src_op._attrs["inputs"])
                    if origin_bmm == target_bmm:
                        if not is_elementwise:
                            self.assertNotEqual(inputs_op[0]._attrs["op"], "permute021")
                    else:
                        self.assertEqual(inputs_op[0]._attrs["op"], "permute021")
            self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

            if is_const:
                X_pt = const_A_data
            else:
                X_pt = torch.randn(_extract_shape(b, shape_A)).cuda().half()
            X_pt_in = X_pt
            if origin_bmm[-3] == "c":
                X_pt_in = torch.permute(X_pt, [0, 2, 1])

            W_pt = torch.randn(_extract_shape(b, shape_B)).cuda().half()
            W_pt_in = W_pt
            if origin_bmm[-2] == "c":
                W_pt_in = torch.permute(W_pt, [0, 2, 1])
            if is_elementwise:
                W_pt_in = torch.add(W_pt_in, W_pt_in)
            Y_pt = torch.matmul(X_pt_in, W_pt_in)
            if strided_output:
                Y_pt = torch.cos(Y_pt)

            inputs = {"input_1": W_pt}
            if not is_const:
                inputs["input_0"] = X_pt

            y = torch.empty([b, M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_permute_bmm_A(self):
        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)

        # const input misaligned on K, permute.
        self._test_permute_bmm_A(
            B,
            [batch_dim, 8, 7],
            [batch_dim, 7, 16],
            "bmm_rrr",
            "bmm_crr",
            is_const=True,
        )
        # const input misaligned on K, permute. [2d broadcast]
        self._test_permute_bmm_A(
            B,
            [8, 7],
            [batch_dim, 7, 16],
            "bmm_rrr",
            "bmm_crr",
            is_const=True,
            test_prefix="2d_broadcast_",
        )
        # non-const input misaligned on K, permute.
        self._test_permute_bmm_A(
            B,
            [batch_dim, 8, 7],
            [batch_dim, 7, 16],
            "bmm_rrr",
            "bmm_crr",
            is_const=False,
        )
        # elementwise input misaligned on K, don't permute.
        self._test_permute_bmm_A(
            B,
            [batch_dim, 8, 7],
            [batch_dim, 7, 16],
            "bmm_rrr",
            "bmm_rrr",
            is_const=False,
            is_elementwise=True,
        )
        # non-const input misaligned on M/N, permute.
        self._test_permute_bmm_A(
            B,
            [batch_dim, 8, 7],
            [batch_dim, 8, 16],
            "bmm_crr",
            "bmm_rrr",
            is_const=False,
            strided_output=False,
        )
        # non-const input misaligned on M/N, less flops, don't permute.
        self._test_permute_bmm_A(
            B,
            [batch_dim, 8, 7],
            [batch_dim, 8, 16],
            "bmm_crr",
            "bmm_crr",
            is_const=False,
            is_elementwise=True,
        )

    def _test_permute_bmm_B(
        self,
        B,
        shape_A,
        shape_B,
        origin_bmm,
        target_bmm,
        is_const,
        is_elementwise=False,
        strided_output=True,
    ):
        M = shape_A[-2] if origin_bmm[-3] == "r" else shape_A[-1]
        N = shape_B[-1] if origin_bmm[-2] == "r" else shape_B[-2]

        for b in B:
            const_A, const_B = None, None
            if is_elementwise:
                const_B = Tensor(
                    shape=shape_B, dtype="float16", name="input_1", is_input=True
                )
                const_B = ops.elementwise(FuncEnum.ADD)(const_B, const_B)
            elif is_const:
                const_B_data = torch.randn(_extract_shape(1, shape_B)).half().cuda()
                const_B = Tensor(
                    shape=_extract_shape(1, shape_B),
                    name="input_1",
                )
                const_B._bind_data(_TorchConstantTensorData(const_B_data))
            bmm_tensor = self._create_permute_bmm_graph(
                shape_A, shape_B, origin_bmm, const_A=const_A, const_B=const_B
            )

            if strided_output:
                output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
            else:
                output = bmm_tensor
            output._attrs["name"] = "output_0"
            output._attrs["is_output"] = True

            # Check value correctness
            target = detect_target()
            module = compile_model(
                output,
                target,
                "./tmp",
                f"alignment_permute_bmm_B_{b}_{origin_bmm}_to_{target_bmm}_{is_const}",
            )

            exist_new_bmm = False
            for tensor in module.debug_sorted_graph:
                src_ops = tensor.src_ops()
                if len(src_ops) == 0:
                    continue
                if not is_elementwise:
                    self.assertEqual(
                        len(src_ops),
                        1,
                        "constructed graph should only have single-source op tensors",
                    )
                src_op = list(tensor.src_ops())[0]
                if src_op._attrs["op"].startswith("bmm"):
                    self.assertEqual(src_op._attrs["op"], target_bmm)
                    exist_new_bmm = True

                    if is_const:
                        continue
                    inputs_op = self._extract_src_op(src_op._attrs["inputs"])
                    if origin_bmm == target_bmm:
                        if not is_elementwise:
                            self.assertNotEqual(inputs_op[1]._attrs["op"], "permute021")
                    else:
                        self.assertEqual(inputs_op[1]._attrs["op"], "permute021")
            self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

            if is_const:
                W_pt = const_B_data
            else:
                W_pt = torch.randn(_extract_shape(b, shape_B)).cuda().half()
            W_pt_in = W_pt
            if origin_bmm[-2] == "c":
                W_pt_in = torch.permute(W_pt, [0, 2, 1])

            X_pt = torch.randn(_extract_shape(b, shape_A)).cuda().half()
            X_pt_in = X_pt
            if origin_bmm[-3] == "c":
                X_pt_in = torch.permute(X_pt, [0, 2, 1])
            if is_elementwise:
                W_pt_in = torch.add(W_pt_in, W_pt_in)
            Y_pt = torch.matmul(X_pt_in, W_pt_in)
            if strided_output:
                Y_pt = torch.cos(Y_pt)

            inputs = {"input_0": X_pt}
            if not is_const:
                inputs["input_1"] = W_pt

            y = torch.empty([b, M, N]).cuda().half()
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_permute_bmm_B(self):
        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)

        # const input misaligned on K, permute.
        self._test_permute_bmm_B(
            B,
            [batch_dim, 7, 8],
            [batch_dim, 12, 7],
            "bmm_ccr",
            "bmm_crr",
            is_const=True,
        )
        # const input misaligned on K, permute. [2d broadcast]
        self._test_permute_bmm_B(
            B,
            [batch_dim, 8, 16],
            [16, 7],
            "bmm_rrr",
            "bmm_rcr",
            is_const=True,
        )
        # non-const input misaligned on K, permute.
        self._test_permute_bmm_B(
            B,
            [batch_dim, 7, 8],
            [batch_dim, 16, 7],
            "bmm_ccr",
            "bmm_crr",
            is_const=False,
        )
        # elementwise input misaligned on K, don't permute.
        self._test_permute_bmm_B(
            B,
            [batch_dim, 7, 8],
            [batch_dim, 16, 7],
            "bmm_ccr",
            "bmm_ccr",
            is_const=False,
            is_elementwise=True,
        )
        # non-const input misaligned on M/N, permute.
        self._test_permute_bmm_B(
            B,
            [batch_dim, 8, 16],
            [batch_dim, 16, 7],
            "bmm_rrr",
            "bmm_rcr",
            is_const=False,
            strided_output=False,
        )
        # non-const input misaligned on M/N, less flop, don't permute.
        self._test_permute_bmm_B(
            B,
            [batch_dim, 8, 16],
            [batch_dim, 16, 7],
            "bmm_rrr",
            "bmm_rrr",
            is_const=False,
            is_elementwise=True,
        )

    def test_permute_bmm_epilogue(self):
        B = [1, 3]
        M = 7
        K = 8
        N = 16
        batch_dim = shape_utils.gen_int_var_min_max(B)
        shape_A = [batch_dim, K, M]
        shape_B = [batch_dim, K, N]
        shape_D = [batch_dim, M, N]

        D = Tensor(shape=shape_D, dtype="float16", name="input_2", is_input=True)

        bmm_tensor = self._create_permute_bmm_graph(shape_A, shape_B, "bmm_crr")
        add_tensor = ops.elementwise(FuncEnum.ADD)(bmm_tensor, D)

        output = ops.elementwise(FuncEnum.COS)(add_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(
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
                self.assertEqual(src_op._attrs["op"], "bmm_rrr_add")
                exist_new_bmm = True

                inputs_op = self._extract_src_op(src_op._attrs["inputs"])
                self.assertEqual(inputs_op[0]._attrs["op"], "permute021")
        self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

        for b in B:
            X_pt = torch.randn(b, K, M).cuda().half()
            W_pt = torch.randn(b, K, N).cuda().half()
            D_pt = torch.randn(b, M, N).cuda().half()
            Y_pt = torch.cos(torch.matmul(torch.permute(X_pt, [0, 2, 1]), W_pt) + D_pt)

            y = torch.empty([b, M, N]).cuda().half()
            module.run_with_tensors(
                {"input_0": X_pt, "input_1": W_pt, "input_2": D_pt}, [y]
            )
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_bmm_pad_special_case(self):
        # We test one case that padding is cheaper than permuting.
        B = [1, 3]
        M = 2
        K = 3
        N = 6
        batch_dim = shape_utils.gen_int_var_min_max(B)
        shape_A = [batch_dim, M, K]
        shape_B = [batch_dim, N, K]

        bmm_tensor = self._create_permute_bmm_graph(shape_A, shape_B, "bmm_rcr")

        output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", "alignment_pad_bmm")

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
        self.assertTrue(exist_new_bmm, "Can't find converted bmm op in graph")

        for b in B:
            X_pt = torch.randn(b, M, K).cuda().half()
            W_pt = torch.randn(b, N, K).cuda().half()
            Y_pt = torch.cos(torch.matmul(X_pt, torch.permute(W_pt, [0, 2, 1])))

            y = torch.empty([b, M, N]).cuda().half()
            module.run_with_tensors({"input_0": X_pt, "input_1": W_pt}, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
