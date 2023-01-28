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
from aitemplate.compiler import compile_model, Model, ops

from aitemplate.compiler.base import (
    _create_host_zero_tensor,
    _TorchConstantTensorData,
    Tensor,
)
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.public import IntImm
from aitemplate.compiler.transform.transform_utils import check_graph_validity

from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)

from parameterized import parameterized


class ConstantFoldingTestCase(unittest.TestCase):
    def _verify_graph(
        self, mod: Model, expected_num_constants: int, expected_num_nodes: int
    ) -> None:
        check_graph_validity(mod.debug_sorted_graph, raiseError=True)
        graph_size = len(mod.debug_sorted_graph)
        self.assertEqual(graph_size, expected_num_nodes)

        num_constants = sum(
            1 for tensor in mod.debug_sorted_graph if tensor._attrs["data"] is not None
        )
        # Make sure the extra constants are deleted.
        self.assertEqual(num_constants, expected_num_constants)

    @parameterized.expand([("float16"), ("float")])
    def test_simple_constant_fold(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by rocm")

        inp0_pt = get_random_torch_tensor((3, 3), dtype)
        inp1_pt = get_random_torch_tensor((3, 3), dtype)
        inp2_pt = get_random_torch_tensor((3, 3), dtype)
        x_pt = inp0_pt * inp1_pt
        y_pt = (inp2_pt + x_pt).flatten()

        inp0_ait = Tensor(shape=(3, 3), dtype=dtype, name="inp0")
        inp0_ait._bind_data(_TorchConstantTensorData(inp0_pt))
        inp1_ait = Tensor(shape=(3, 3), dtype=dtype, name="inp1")
        inp1_ait._bind_data(_TorchConstantTensorData(inp1_pt))
        inp2_ait = Tensor(shape=[3, 3], dtype=dtype, name="inp2", is_input=True)

        x_ait = ops.elementwise(FuncEnum.MUL)(inp0_ait, inp1_ait)
        # prevent mul/add fusion. If the ops get fused, then inp2_ait will be
        # an input to the fused op, which will prevent constant folding.
        x_view = ops.flatten()(x_ait)
        inp2_view = ops.flatten()(inp2_ait)
        y_ait = ops.elementwise(FuncEnum.ADD)(inp2_view, x_view)
        y_ait._attrs["name"] = "y"
        y_ait._attrs["is_output"] = True

        mod = compile_model(
            y_ait, target, "./tmp", f"test_constant_folding_simple_{dtype}"
        )

        y = get_torch_empty_tensor((9,), dtype)
        mod.run_with_tensors({"inp2": inp2_pt}, {"y": y})
        self.assertTrue(torch.equal(y, y_pt))

        # Make sure we eliminated the first elementwise op. We start with 7
        # tensors, eliminate the elementwise op + its 2 inputs + two flattens,
        # and add one constant, so the total size should be 3.
        self._verify_graph(mod, expected_num_constants=1, expected_num_nodes=3)

    @parameterized.expand([("float16"), ("float")])
    def test_pad_constant_weight(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M, N, K = 16, 32, 3
        w_pt = get_random_torch_tensor((K, N), dtype)
        weight_data = _TorchConstantTensorData(w_pt)
        input_0 = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[K, N], dtype=dtype, name="weight")
        W._bind_data(weight_data)
        Y = ops.gemm_rrr()(input_0, W)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        mod = compile_model(Y, target, "./tmp", f"test_pad_constant_weight_{dtype}")

        input_0_pt = get_random_torch_tensor((M, K), dtype)
        y_pt = torch.matmul(input_0_pt, w_pt)

        y = get_torch_empty_tensor((M, N), dtype)
        mod.run_with_tensors({"input_0": input_0_pt}, {"y": y})

        torch.testing.assert_close(y, y_pt, atol=1e-1, rtol=1e-1)

        # For float16 inputs, the apply_padding graph pass will add padding to
        # both the input and the weight in this case with concatenate().
        # The concatenate for the weight will be folded, so we will be left with
        # 2 constants.
        if dtype == "float16":
            expected_num_constants = 2
            expected_num_nodes = 5
        elif dtype == "float":
            # Gemm ops with float inputs do not have any alignment requirements,
            # so the apply_padding pass will not add any padding constants.
            # The final graph only contains the original "weight" constant tensor.
            expected_num_constants = 1
            expected_num_nodes = 3
        else:
            raise RuntimeError(f"invalid {dtype=}")
        self._verify_graph(
            mod,
            expected_num_constants=expected_num_constants,
            expected_num_nodes=expected_num_nodes,
        )

    @parameterized.expand([("float16"), ("float")])
    def test_fold_long_chain(self, dtype):
        target = detect_target()
        if dtype == "float" and (target.name == "rocm" or int(target._arch) < 80):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")
        M, N, K = 16, 32, 3
        w1_pt = get_random_torch_tensor((K, N), dtype)
        w1_data = _TorchConstantTensorData(w1_pt)

        w2_pt = get_random_torch_tensor((K, N), dtype)
        w2_data = _TorchConstantTensorData(w2_pt)

        w3_pt = w1_pt * w2_pt
        x_pt = get_random_torch_tensor((M, K), dtype)
        x_pt_data = _TorchConstantTensorData(x_pt)

        y_pt = torch.matmul(x_pt, w3_pt)
        w4_pt = get_random_torch_tensor((M, N), dtype)
        w4_data = _TorchConstantTensorData(w4_pt)
        z_pt = y_pt * w4_pt

        w1_ait = Tensor(shape=[K, N], dtype=dtype, name="w1")
        w1_ait._bind_data(w1_data)
        w2_ait = Tensor(shape=[K, N], dtype=dtype, name="w2")
        w2_ait._bind_data(w2_data)
        w3_ait = ops.elementwise(FuncEnum.MUL)(w1_ait, w2_ait)
        x_ait = Tensor(shape=[M, K], dtype=dtype, name="x")
        x_ait._bind_data(x_pt_data)
        y_ait = ops.gemm_rrr()(x_ait, w3_ait)
        w4_ait = Tensor(shape=[M, N], dtype=dtype, name="w4")
        w4_ait._bind_data(w4_data)
        z_ait = ops.elementwise(FuncEnum.MUL)(y_ait, w4_ait)
        z_ait._attrs["name"] = "z"
        z_ait._attrs["is_output"] = True

        target = detect_target()
        mod = compile_model(z_ait, target, "./tmp", f"test_fold_long_chain_{dtype}")

        z = get_torch_empty_tensor((M, N), dtype)
        mod.run_with_tensors({}, {"z": z})

        torch.testing.assert_close(z, z_pt, atol=1e-1, rtol=1e-1)

        # The entire graph is turned into a constant.
        self._verify_graph(mod, expected_num_constants=1, expected_num_nodes=1)

    @parameterized.expand([("float16"), ("float")])
    def test_constant_folding_through_views(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by rocm")

        inp0_pt = get_random_torch_tensor((3, 3), dtype)
        inp1_pt = get_random_torch_tensor((3, 3), dtype)
        y_pt = (inp0_pt * inp1_pt).flatten()

        inp0_ait = Tensor(shape=(3, 3), dtype=dtype, name="inp0")
        inp0_ait._bind_data(_TorchConstantTensorData(inp0_pt))
        inp1_ait = Tensor(shape=(3, 3), dtype=dtype, name="inp1")
        inp1_ait._bind_data(_TorchConstantTensorData(inp1_pt))
        inp0_view = ops.flatten()(inp0_ait)
        inp1_view = ops.flatten()(inp1_ait)
        y_ait = ops.elementwise(FuncEnum.MUL)(inp0_view, inp1_view)
        y_ait._attrs["name"] = "y"
        y_ait._attrs["is_output"] = True

        mod = compile_model(
            y_ait, target, "./tmp", f"test_constant_folding_through_views_{dtype}"
        )

        y = get_torch_empty_tensor((9,), dtype)
        mod.run_with_tensors({}, {"y": y})
        self.assertTrue(torch.equal(y, y_pt))

        # The entire graph is eliminated.
        self._verify_graph(mod, expected_num_constants=1, expected_num_nodes=1)

    @parameterized.expand([("float16"), ("float")])
    def test_late_binding(self, dtype):
        target = detect_target()
        if dtype == "float" and (target.name == "rocm" or int(target._arch) < 80):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        # Test binding constants through compile_model
        M, N, K = 16, 32, 3
        w1_pt = get_random_torch_tensor((K, N), dtype)

        w2_pt = get_random_torch_tensor((K, N), dtype)

        w3_pt = w1_pt * w2_pt
        x_pt = get_random_torch_tensor((M, K), dtype)

        y_pt = torch.matmul(x_pt, w3_pt)
        w4_pt = get_random_torch_tensor((M, N), dtype)
        z_pt = y_pt * w4_pt

        w1_ait = Tensor(shape=[K, N], dtype=dtype, name="w1")
        w2_ait = Tensor(shape=[K, N], dtype=dtype, name="w2")
        w3_ait = ops.elementwise(FuncEnum.MUL)(w1_ait, w2_ait)
        x_ait = Tensor(shape=[M, K], dtype=dtype, name="x")
        y_ait = ops.gemm_rrr()(x_ait, w3_ait)
        w4_ait = Tensor(shape=[M, N], dtype=dtype, name="w4")
        z_ait = ops.elementwise(FuncEnum.MUL)(y_ait, w4_ait)
        z_ait._attrs["name"] = "z"
        z_ait._attrs["is_output"] = True

        mod = compile_model(
            z_ait,
            target,
            "./tmp",
            f"test_late_binding_{dtype}",
            constants={"w1": w1_pt, "w2": w2_pt, "x": x_pt, "w4": w4_pt},
        )

        z = get_torch_empty_tensor((M, N), dtype)
        mod.run_with_tensors({}, {"z": z})

        torch.testing.assert_close(z, z_pt, atol=1e-1, rtol=1e-1)

        # The entire graph is turned into a constant.
        self._verify_graph(mod, expected_num_constants=1, expected_num_nodes=1)

    def test_late_binding_error_constant_already_bound(self):
        dtype = "float16"

        N, K = IntImm(16), IntImm(32)
        w1_ait = _create_host_zero_tensor(shape=[K, N], name="w1", dtype=dtype)
        w2_ait = Tensor(shape=[K, N], dtype=dtype, name="w2")
        y_ait = ops.elementwise(FuncEnum.MUL)(w1_ait, w2_ait)
        y_ait._attrs["name"] = "y"
        y_ait._attrs["is_output"] = True

        torch_shape = (K.value(), N.value())
        target = detect_target()
        with self.assertRaisesRegex(ValueError, "Tensor w1 is already bound!"):
            compile_model(
                y_ait,
                target,
                "./tmp",
                "test_late_binding",
                constants={
                    "w1": get_random_torch_tensor(torch_shape, dtype),
                    "w2": get_random_torch_tensor(torch_shape, dtype),
                },
            )

    def test_late_binding_error_cannot_bind_input(self):
        dtype = "float16"

        N, K = IntImm(16), IntImm(32)
        w1_ait = Tensor(shape=[K, N], dtype=dtype, name="w1", is_input=True)
        w2_ait = Tensor(shape=[K, N], dtype=dtype, name="w2")
        y_ait = ops.elementwise(FuncEnum.MUL)(w1_ait, w2_ait)
        y_ait._attrs["name"] = "y"
        y_ait._attrs["is_output"] = True

        torch_shape = (K.value(), N.value())
        target = detect_target()
        with self.assertRaisesRegex(ValueError, "Cannot bind input tensor w1"):
            compile_model(
                y_ait,
                target,
                "./tmp",
                "test_late_binding",
                constants={
                    "w1": get_random_torch_tensor(torch_shape, dtype),
                    "w2": get_random_torch_tensor(torch_shape, dtype),
                },
            )

    def test_late_binding_error_cannot_bind_non_constant(self):
        dtype = "float16"

        N, K = IntImm(16), IntImm(32)
        w1_ait = Tensor(shape=[K, N], dtype=dtype, name="w1")
        w2_ait = Tensor(shape=[K, N], dtype=dtype, name="w2")
        y_ait = ops.elementwise(FuncEnum.MUL)(w1_ait, w2_ait)
        y_ait._attrs["name"] = "y"
        y_ait._attrs["is_output"] = True

        torch_shape = (K.value(), N.value())
        target = detect_target()
        with self.assertRaisesRegex(ValueError, "Cannot bind non-constant tensor y"):
            compile_model(
                y_ait,
                target,
                "./tmp",
                "test_late_binding",
                constants={
                    "w1": get_random_torch_tensor(torch_shape, dtype),
                    "w2": get_random_torch_tensor(torch_shape, dtype),
                    "y": get_random_torch_tensor(torch_shape, dtype),
                },
            )

    def test_late_binding_fails_wrong_dtype(self):
        dtype = "float16"

        w1_ait = Tensor(shape=[1], name="w1", dtype=dtype)
        y = ops.elementwise(FuncEnum.MUL)(w1_ait, w1_ait)
        y._attrs["name"] = "y"
        y._attrs["is_output"] = True

        wrong_inputs = (
            torch.randn((1,)),
            torch.zeros((1,)).int(),
            torch.zeros((1,)).long(),
        )

        for w1_pt in wrong_inputs:
            with self.assertRaisesRegex(
                ValueError,
                r"data's dtype did not match: expected float16, got .*",
            ):
                target = detect_target()
                compile_model(
                    y,
                    target,
                    "./tmp",
                    "test_late_binding_fails_wrong_dtype",
                    constants={"w1": w1_pt},
                )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
