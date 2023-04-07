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
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)
from aitemplate.utils import shape_utils

from parameterized import parameterized


def custom_name_func_with_testname(testcase_func, param_num, param):
    return "%s_%s_sm80" % (
        testcase_func.__name__[:-5],
        param.args[-2],
    )


def custom_name_func_with_funcname(testcase_func, param_num, param):
    return "%s_%s_sm80" % (
        testcase_func.__name__[:-5],
        str(param.args[0].__name__),
    )


class FusePermuteBmmCase(unittest.TestCase):
    def _create_permute_bmm_graph(
        self, A_shape, B_shape, bmm_type, permA, permB, dtype, bias_shape=None
    ):
        OP = getattr(ops, bmm_type, None)
        assert OP is not None

        A = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        B = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        X = A
        W = B
        if permA:
            A = ops.permute021()(A)
        if permB:
            B = ops.permute021()(B)
        inputs = [A, B]
        if bias_shape is not None:
            inputs.append(
                Tensor(shape=bias_shape, dtype=dtype, name="input_2", is_input=True)
            )

        Y = OP()(*inputs)
        Y._attrs["name"] = "target_bmm_tensor"
        return X, W, Y

    def _test_missing_alignment_bmm(
        self, A_shape, B_shape, bmm_type, permA, permB, testname, dtype="float16"
    ):
        X, W, bmm_tensor = self._create_permute_bmm_graph(
            A_shape, B_shape, bmm_type, permA, permB, dtype
        )
        output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        if dtype == "float":
            expected_bmm_type = list(bmm_type)
            if permA:
                if expected_bmm_type[-3] == "c":
                    expected_bmm_type[-3] = "r"
                else:
                    expected_bmm_type[-3] = "c"
            if permB:
                if expected_bmm_type[-2] == "c":
                    expected_bmm_type[-2] = "r"
                else:
                    expected_bmm_type[-2] = "c"
            expected_bmm_type = "".join(expected_bmm_type)
        else:
            expected_bmm_type = bmm_type
        found_tensor = False
        for tensor in module.debug_sorted_graph:
            src_ops = tensor.src_ops()
            if len(src_ops) == 0:
                continue
            assert (
                len(src_ops) == 1
            ), "constructed graph should only have single-source op tensors"
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"].startswith("bmm"):
                found_tensor = True
                self.assertEqual(src_op._attrs["op"], expected_bmm_type)
        self.assertTrue(found_tensor)

    def test_misalign_a_bmm(self):
        self._test_missing_alignment_bmm(
            [2, 4, 7], [2, 7, 8], "bmm_crr", True, False, "bmm_crr_misalign_a"
        )
        self._test_missing_alignment_bmm(
            [2, 4, 7], [2, 8, 4], "bmm_rcr", True, False, "bmm_rcr_misalign_a"
        )
        self._test_missing_alignment_bmm(
            [2, 4, 7], [2, 4, 8], "bmm_rrr", True, False, "bmm_rrr_misalign_a"
        )

    def test_misalign_b_bmm(self):
        self._test_missing_alignment_bmm(
            [2, 8, 4], [2, 8, 7], "bmm_ccr", False, True, "bmm_ccr_misalign_b"
        )
        self._test_missing_alignment_bmm(
            [2, 7, 8], [2, 8, 7], "bmm_crr", False, True, "bmm_crr_misalign_b"
        )
        self._test_missing_alignment_bmm(
            [2, 4, 8], [2, 8, 7], "bmm_rcr", False, True, "bmm_rcr_misalign_b"
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @parameterized.expand(
        [
            (
                _test_missing_alignment_bmm,
                [2, 4, 7],
                [2, 7, 8],
                "bmm_crr",
                True,
                False,
                "bmm_crr_misalign_a",
                "float",
            ),
            (
                _test_missing_alignment_bmm,
                [2, 4, 7],
                [2, 8, 4],
                "bmm_rcr",
                True,
                False,
                "bmm_rcr_misalign_a",
                "float",
            ),
            (
                _test_missing_alignment_bmm,
                [2, 4, 7],
                [2, 4, 8],
                "bmm_rrr",
                True,
                False,
                "bmm_rrr_misalign_a",
                "float",
            ),
            (
                _test_missing_alignment_bmm,
                [2, 8, 4],
                [2, 8, 7],
                "bmm_ccr",
                False,
                True,
                "bmm_ccr_misalign_b",
                "float",
            ),
            (
                _test_missing_alignment_bmm,
                [2, 7, 8],
                [2, 8, 7],
                "bmm_crr",
                False,
                True,
                "bmm_crr_misalign_b",
                "float",
            ),
            (
                _test_missing_alignment_bmm,
                [2, 4, 8],
                [2, 8, 7],
                "bmm_rcr",
                False,
                True,
                "bmm_rcr_misalign_b",
                "float",
            ),
        ],
        name_func=custom_name_func_with_testname,
    )
    def test_misalign_bmm_float_sm80(
        self, func, A_shape, B_shape, bmm_type, permA, permB, testname, dtype
    ):
        func(self, A_shape, B_shape, bmm_type, permA, permB, testname, dtype)

    def _test_permute_bmm(
        self,
        B,
        A_shape,
        B_shape,
        original_bmm,
        new_bmm,
        testname,
        dtype="float16",
        bias_shape=None,
    ):
        new_layout = new_bmm[-3:]
        if new_layout[0] == "r":
            M, K = A_shape[-2:]
        else:
            K, M = A_shape[-2:]

        if new_layout[1] == "r":
            N = B_shape[-1]
        else:
            N = B_shape[-2]

        permA = original_bmm[-3] != new_bmm[-3]
        permB = original_bmm[-2] != new_bmm[-2]

        if bias_shape is not None:
            assert original_bmm.startswith("gemm")
            assert new_bmm.startswith("bmm")
            original_bmm += "_bias"
            new_bmm += "_add"

        X, W, bmm_tensor = self._create_permute_bmm_graph(
            A_shape,
            B_shape,
            original_bmm,
            permA,
            permB,
            dtype,
            bias_shape=bias_shape,
        )

        output = ops.elementwise(FuncEnum.COS)(bmm_tensor)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        # Check value correctness
        target = detect_target()
        module = compile_model(output, target, "./tmp", testname)

        # Due to massive rewriting of alignment/padding, we check whether we removed the old bmm with new one instead.
        exist_new_bmm = False
        for tensor in module.debug_sorted_graph:
            src_ops = tensor.src_ops()
            if len(src_ops) == 0:
                continue
            assert (
                len(src_ops) == 1
            ), "constructed graph should only have single-source op tensors"
            src_op = list(tensor.src_ops())[0]
            assert src_op._attrs["op"] != original_bmm

            if src_op._attrs["op"] == new_bmm:
                exist_new_bmm = True

        assert exist_new_bmm, "Can't find converted bmm op in graph"

        for b in B:
            if len(A_shape) > 2:
                X_pt = get_random_torch_tensor([b, M, K], dtype)
            else:
                X_pt = get_random_torch_tensor([M, K], dtype)

            if len(B_shape) > 2:
                W_pt = get_random_torch_tensor([b, K, N], dtype)
            else:
                W_pt = get_random_torch_tensor([K, N], dtype)

            Y_pt = torch.matmul(X_pt, W_pt)

            if bias_shape is not None:
                bias_pt = get_random_torch_tensor(bias_shape[0], dtype)
                Y_pt += bias_pt

            Y_pt = torch.cos(Y_pt)

            if new_layout[0] == "c":
                perm = (0, 2, 1) if len(A_shape) > 2 else (1, 0)
                X_pt = X_pt.permute(perm).contiguous()
            if new_layout[1] == "c":
                perm = (0, 2, 1) if len(B_shape) > 2 else (1, 0)
                W_pt = W_pt.permute(perm).contiguous()

            # We currently only have row-major outputs.
            y = get_torch_empty_tensor([b, M, N], dtype)

            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0, 0] if bias_shape is None else [0, 0, 0]
            inputs[input_name_to_index["input_0"]] = X_pt
            inputs[input_name_to_index["input_1"]] = W_pt
            if bias_shape is not None:
                inputs[input_name_to_index["input_2"]] = bias_pt

            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    def test_ccr_to_rrr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 2, 4], [batch_dim, 4, 8], "bmm_ccr", "bmm_rrr", "ccr_to_rrr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [batch_dim, 5, 7],
            "bmm_ccr",
            "bmm_rrr",
            "ccr_to_rrr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [batch_dim, 4, 8],
            "bmm_ccr",
            "bmm_rrr",
            "ccr_to_rrr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_ccr_to_rrr_float_sm80(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [batch_dim, 5, 7],
            "bmm_ccr",
            "bmm_rrr",
            "ccr_to_rrr_need_align_float",
            dtype="float",
        )

    def test_ccr_to_crr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 4, 2], [batch_dim, 4, 8], "bmm_ccr", "bmm_crr", "ccr_to_crr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 5, 3],
            [batch_dim, 5, 7],
            "bmm_ccr",
            "bmm_crr",
            "ccr_to_crr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [batch_dim, 4, 8],
            "bmm_ccr",
            "bmm_crr",
            "ccr_to_crr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_ccr_to_crr_float_sm80(self):
        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [batch_dim, 4, 8],
            "bmm_ccr",
            "bmm_crr",
            "ccr_to_crr_dynamic_float",
            dtype="float",
        )

    def test_ccr_to_rcr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 2, 4], [batch_dim, 8, 4], "bmm_ccr", "bmm_rcr", "ccr_to_rcr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [batch_dim, 7, 5],
            "bmm_ccr",
            "bmm_rcr",
            "ccr_to_rcr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [batch_dim, 8, 4],
            "bmm_ccr",
            "bmm_rcr",
            "ccr_to_rcr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_ccr_to_rcr_float_sm80(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [batch_dim, 8, 4],
            "bmm_ccr",
            "bmm_rcr",
            "ccr_to_rcr_float",
            dtype="float",
        )

    def test_crr_to_ccr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 4, 2], [batch_dim, 8, 4], "bmm_crr", "bmm_ccr", "crr_to_ccr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 5, 3],
            [batch_dim, 7, 5],
            "bmm_crr",
            "bmm_ccr",
            "crr_to_ccr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [batch_dim, 8, 4],
            "bmm_crr",
            "bmm_ccr",
            "crr_to_ccr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_crr_to_ccr_float_sm80(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [batch_dim, 8, 4],
            "bmm_crr",
            "bmm_ccr",
            "crr_to_ccr_float",
            dtype="float",
        )

    def test_crr_to_rrr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 2, 4], [batch_dim, 4, 8], "bmm_crr", "bmm_rrr", "crr_to_rrr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [batch_dim, 5, 7],
            "bmm_crr",
            "bmm_rrr",
            "crr_to_rrr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [batch_dim, 4, 8],
            "bmm_crr",
            "bmm_rrr",
            "crr_to_rrr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_crr_to_rrr_float_sm80(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [batch_dim, 5, 7],
            "bmm_crr",
            "bmm_rrr",
            "crr_to_rrr_need_align_float",
            dtype="float",
        )

    def test_rcr_to_ccr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 4, 2], [batch_dim, 8, 4], "bmm_rcr", "bmm_ccr", "rcr_to_ccr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 5, 3],
            [batch_dim, 7, 5],
            "bmm_rcr",
            "bmm_ccr",
            "rcr_to_ccr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [batch_dim, 8, 4],
            "bmm_rcr",
            "bmm_ccr",
            "rcr_to_ccr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_rcr_to_ccr_float_sm80(self):
        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [batch_dim, 8, 4],
            "bmm_rcr",
            "bmm_ccr",
            "rcr_to_ccr_dynamic_float",
            dtype="float",
        )

    def test_rcr_to_rrr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 2, 4], [batch_dim, 4, 8], "bmm_rcr", "bmm_rrr", "rcr_to_rrr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [batch_dim, 5, 7],
            "bmm_rcr",
            "bmm_rrr",
            "rcr_to_rrr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [batch_dim, 4, 8],
            "bmm_rcr",
            "bmm_rrr",
            "rcr_to_rrr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_rcr_to_rrr_float_sm80(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [batch_dim, 4, 8],
            "bmm_rcr",
            "bmm_rrr",
            "rcr_to_rrr_float",
            dtype="float",
        )

    def test_rrr_to_crr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 4, 2], [batch_dim, 4, 8], "bmm_rrr", "bmm_crr", "rrr_to_crr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 5, 3],
            [batch_dim, 5, 7],
            "bmm_rrr",
            "bmm_crr",
            "rrr_to_crr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [batch_dim, 4, 8],
            "bmm_rrr",
            "bmm_crr",
            "rrr_to_crr_dynamic",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_rrr_to_crr_float_sm80(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 5, 3],
            [batch_dim, 5, 7],
            "bmm_rrr",
            "bmm_crr",
            "rrr_to_crr_need_align_float",
            dtype="float",
        )

    def test_rrr_to_rcr(self):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 2, 4], [batch_dim, 8, 4], "bmm_rrr", "bmm_rcr", "rrr_to_rcr"
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [batch_dim, 7, 5],
            "bmm_rrr",
            "bmm_rcr",
            "rrr_to_rcr_need_align",
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B, [batch_dim, 2, 4], [batch_dim, 8, 4], "bmm_rrr", "bmm_rcr", "rrr_to_rcr"
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_rrr_to_rcr_float_sm80(self):
        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [batch_dim, 8, 4],
            "bmm_rrr",
            "bmm_rcr",
            "rrr_to_rcr_float",
            dtype="float",
        )

    def _test_gemm_broadcast_rcr_to_ccr(self, test_bias, dtype="float16"):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [8, 4],
            "gemm_rcr",
            "bmm_ccr",
            f"rcr_to_ccr_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[8] if test_bias else None,
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 5, 3],
            [7, 5],
            "gemm_rcr",
            "bmm_ccr",
            f"rcr_to_ccr_need_align_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[7] if test_bias else None,
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [8, 4],
            "gemm_rcr",
            "bmm_ccr",
            f"rcr_to_ccr_dynamic_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[8] if test_bias else None,
        )

    def _test_gemm_broadcast_rcr_to_rrr(self, test_bias, dtype="float16"):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [4, 8],
            "gemm_rcr",
            "bmm_rrr",
            f"rcr_to_rrr_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[8] if test_bias else None,
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 3, 5],
            [5, 7],
            "gemm_rcr",
            "bmm_rrr",
            f"rcr_to_rrr_need_align_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[7] if test_bias else None,
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 2, 4],
            [4, 8],
            "gemm_rcr",
            "bmm_rrr",
            f"rcr_to_rrr_dynamic_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[8] if test_bias else None,
        )

    def _test_gemm_broadcast_rrr_to_crr(self, test_bias, dtype="float16"):
        B = [1]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [4, 8],
            "gemm_rrr",
            "bmm_crr",
            f"rrr_to_crr_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[8] if test_bias else None,
        )
        self._test_permute_bmm(
            B,
            [batch_dim, 5, 3],
            [5, 7],
            "gemm_rrr",
            "bmm_crr",
            f"rrr_to_crr_need_align_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[7] if test_bias else None,
        )

        B = [1, 3]
        batch_dim = shape_utils.gen_int_var_min_max(B)
        self._test_permute_bmm(
            B,
            [batch_dim, 4, 2],
            [4, 8],
            "gemm_rrr",
            "bmm_crr",
            f"rrr_to_crr_dynamic_gemm_broadcast_b_{dtype}",
            dtype,
            bias_shape=[8] if test_bias else None,
        )

    def test_gemm_broadcast_rcr_to_ccr(self):
        self._test_gemm_broadcast_rcr_to_ccr(True)
        self._test_gemm_broadcast_rcr_to_ccr(False)

    def test_gemm_broadcast_rrr_to_crr(self):
        self._test_gemm_broadcast_rrr_to_crr(True)
        self._test_gemm_broadcast_rrr_to_crr(False)

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    @parameterized.expand(
        [
            (
                _test_gemm_broadcast_rcr_to_ccr,
                True,
                "float",
            ),
            (
                _test_gemm_broadcast_rrr_to_crr,
                False,
                "float",
            ),
        ],
        name_func=custom_name_func_with_funcname,
    )
    def test_gemm_broadcast_float_sm80(self, func, test_bias, dtype):
        func(self, test_bias, dtype)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_permute_multiple_consumer(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        A_shape = [2, 8, 4]
        B_shape = [2, 8, 8]

        A = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        B1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)

        permA = ops.permute021()(A)

        C1 = ops.bmm_rrr()(permA, B1)
        C2 = ops.elementwise(FuncEnum.COS)(permA)

        output = ops.concatenate()((C1, C2), dim=0)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        module = compile_model(
            output, target, "./tmp", f"permute_multiple_consumer_{dtype}"
        )

        graph = module.debug_sorted_graph
        bmm_tensors = 0
        for tensor in graph:
            src_ops = tensor.src_ops()
            if len(src_ops) != 2:
                continue
            src_op = list(tensor.src_ops())[0]
            if src_op._attrs["op"].startswith("bmm"):
                bmm_tensors += 1
                self.assertEqual(src_op._attrs["op"], "bmm_crr")
        self.assertEqual(bmm_tensors, 1)

        A_pt = get_random_torch_tensor(A_shape, dtype)
        AT_pt = A_pt.permute((0, 2, 1))
        B1_pt = get_random_torch_tensor(B_shape, dtype)

        C1_pt = torch.bmm(AT_pt, B1_pt)
        C2_pt = torch.cos(AT_pt)

        Y_pt = torch.concat((C1_pt, C2_pt), dim=0)

        y = get_torch_empty_tensor([4, 4, 8], dtype)
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0, 0]
        inputs[input_name_to_index["input_0"]] = A_pt
        inputs[input_name_to_index["input_1"]] = B1_pt

        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_permute_multiple_only_bmm_consumer(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        A_shape = [2, 8, 4]
        B_shape = [2, 8, 8]

        A = Tensor(shape=A_shape, dtype=dtype, name="input_0", is_input=True)
        B1 = Tensor(shape=B_shape, dtype=dtype, name="input_1", is_input=True)
        B2 = Tensor(shape=B_shape, dtype=dtype, name="input_2", is_input=True)

        permA = ops.permute021()(A)

        C1 = ops.bmm_rrr()(permA, B1)
        C2 = ops.bmm_rrr()(permA, B2)

        output = ops.concatenate()((C1, C2), dim=0)
        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        module = compile_model(
            output, target, "./tmp", f"permute_multiple_bmm_consumer_{dtype}"
        )

        graph = module.debug_sorted_graph
        bmm_tensors = 0
        for tensor in graph:
            src_ops = tensor.src_ops()
            if len(src_ops) != 2:
                continue
            for src_op in list(tensor.src_ops()):
                # All permutes should've be gone.
                self.assertFalse(src_op._attrs["op"].startswith("permute"))
                if src_op._attrs["op"].startswith("bmm"):
                    bmm_tensors += 1
                    self.assertEqual(src_op._attrs["op"], "bmm_crr")
        self.assertEqual(bmm_tensors, 2)

        A_pt = get_random_torch_tensor(A_shape, dtype)
        AT_pt = A_pt.permute((0, 2, 1))
        B1_pt = get_random_torch_tensor(B_shape, dtype)
        B2_pt = get_random_torch_tensor(B_shape, dtype)

        C1_pt = torch.bmm(AT_pt, B1_pt)
        C2_pt = torch.bmm(AT_pt, B2_pt)

        Y_pt = torch.concat((C1_pt, C2_pt), dim=0)

        y = get_torch_empty_tensor([4, 4, 8], dtype)
        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0, 0, 0]
        inputs[input_name_to_index["input_0"]] = A_pt
        inputs[input_name_to_index["input_1"]] = B1_pt
        inputs[input_name_to_index["input_2"]] = B2_pt

        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))


filter_test_cases_by_test_env(FusePermuteBmmCase)
if __name__ == "__main__":
    unittest.main()
