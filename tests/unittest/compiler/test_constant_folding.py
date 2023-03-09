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
import itertools
import unittest

import torch
from aitemplate.compiler import compile_model, Model, ops

from aitemplate.compiler.base import _create_host_zero_tensor, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.public import IntImm
from aitemplate.compiler.transform.transform_utils import check_graph_validity

from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    TestEnv,
)

from parameterized import parameterized


class ConstantFoldingTestCase(unittest.TestCase):
    def _verify_graph(
        self, mod: Model, expected_num_constants: int, expected_num_nodes: int
    ) -> None:
        check_graph_validity(mod.debug_sorted_graph, raiseError=True)
        graph_size = len(mod.debug_sorted_graph)
        self.assertEqual(graph_size, expected_num_nodes)

    @parameterized.expand([("float16")])
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
        inp1_ait = Tensor(shape=(3, 3), dtype=dtype, name="inp1")
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
        mod.set_constant_with_tensor("inp0", inp0_pt)
        mod.set_constant_with_tensor("inp1", inp1_pt)
        mod.fold_constants()

        y = get_torch_empty_tensor((9,), dtype)
        mod.run_with_tensors({"inp2": inp2_pt}, {"y": y})
        self.assertTrue(torch.equal(y, y_pt))

        # Make sure we eliminated the first elementwise op. We start with 7
        # tensors, eliminate the elementwise op + its 2 inputs + two flattens,
        # and add one constant, so the total size should be 3.
        self._verify_graph(mod, expected_num_constants=1, expected_num_nodes=3)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_pad_constant_weight(self, dtype):
        target = detect_target()
        if dtype == "float" and (int(target._arch) < 80 or target.name == "rocm"):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")

        M, N, K = 16, 32, 3
        w_pt = get_random_torch_tensor((K, N), dtype)
        input_0 = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[K, N], dtype=dtype, name="weight")
        Y = ops.gemm_rrr()(input_0, W)
        Y._attrs["name"] = "y"
        Y._attrs["is_output"] = True

        mod = compile_model(Y, target, "./tmp", f"test_pad_constant_weight_{dtype}")
        mod.set_constant_with_tensor("weight", w_pt)
        mod.fold_constants()

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

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_fold_long_chain(self, dtype):
        target = detect_target()
        if dtype == "float" and (target.name == "rocm" or int(target._arch) < 80):
            self.skipTest("gemm with float tensors requires CUDA sm >= 80")
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

        target = detect_target()
        mod = compile_model(z_ait, target, "./tmp", f"test_fold_long_chain_{dtype}")
        mod.set_constant_with_tensor("w1", w1_pt)
        mod.set_constant_with_tensor("w2", w2_pt)
        mod.set_constant_with_tensor("x", x_pt)
        mod.set_constant_with_tensor("w4", w4_pt)
        mod.fold_constants()

        z = get_torch_empty_tensor((M, N), dtype)
        mod.run_with_tensors({}, {"z": z})

        torch.testing.assert_close(z, z_pt, atol=1e-1, rtol=1e-1)

        # The entire graph is turned into a constant.
        self._verify_graph(mod, expected_num_constants=1, expected_num_nodes=1)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
    def test_constant_folding_through_views(self, dtype):
        target = detect_target()
        if dtype == "float" and target.name == "rocm":
            self.skipTest("float tensors not supported by rocm")

        inp0_pt = get_random_torch_tensor((3, 3), dtype)
        inp1_pt = get_random_torch_tensor((3, 3), dtype)
        y_pt = (inp0_pt * inp1_pt).flatten()

        inp0_ait = Tensor(shape=(3, 3), dtype=dtype, name="inp0")
        inp1_ait = Tensor(shape=(3, 3), dtype=dtype, name="inp1")
        inp0_view = ops.flatten()(inp0_ait)
        inp1_view = ops.flatten()(inp1_ait)
        y_ait = ops.elementwise(FuncEnum.MUL)(inp0_view, inp1_view)
        y_ait._attrs["name"] = "y"
        y_ait._attrs["is_output"] = True

        mod = compile_model(
            y_ait, target, "./tmp", f"test_constant_folding_through_views_{dtype}"
        )
        mod.set_constant_with_tensor("inp0", inp0_pt)
        mod.set_constant_with_tensor("inp1", inp1_pt)
        mod.fold_constants()

        y = get_torch_empty_tensor((9,), dtype)
        mod.run_with_tensors({}, {"y": y})
        self.assertTrue(torch.equal(y, y_pt))

        # The entire graph is eliminated.
        self._verify_graph(mod, expected_num_constants=1, expected_num_nodes=1)

    @parameterized.expand(
        filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("float")],
            }
        )
    )
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
        )
        mod.set_constant_with_tensor("w1", w1_pt)
        mod.set_constant_with_tensor("w2", w2_pt)
        mod.set_constant_with_tensor("x", x_pt)
        mod.set_constant_with_tensor("w4", w4_pt)
        mod.fold_constants()

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

    def test_constant_folding_manual_call(self):
        dtype = "float16"

        N, K = IntImm(16), IntImm(32)
        w1_ait = Tensor(shape=[K, N], dtype=dtype, name="w1")
        w2_ait = Tensor(shape=[K, N], dtype=dtype, name="w2")
        y_ait = ops.elementwise(FuncEnum.MUL)(w1_ait, w2_ait)
        y_ait._attrs["name"] = "y"
        y_ait._attrs["is_output"] = True

        shape = (K.value(), N.value())
        w1_pt = get_random_torch_tensor(shape, dtype)
        w2_pt = get_random_torch_tensor(shape, dtype)
        y_pt = w1_pt * w2_pt
        y = torch.empty_like(y_pt)

        with compile_model(
            y_ait, detect_target(), "./tmp", "test_constant_folding_manual_call"
        ) as mod:
            # Unset constants
            self.assertRaises(RuntimeError, mod.run_with_tensors, {}, [y])
            self.assertRaises(RuntimeError, mod.fold_constants)

            mod.set_many_constants_with_tensors({"w1": w1_pt, "w2": w2_pt})
            mod.fold_constants()
            mod.run_with_tensors({}, [y])
            self.assertTrue(torch.equal(y_pt, y))

    def test_constant_folding_mixed_usage(self):
        """
        Test a mix of all the ways to use constants:
        - Unbound constant that is not folded
        - Unbound constant folding input
        - Bound constant folding input
        - Bound constant that is not folded
        """
        dtype = "float16"

        N, K = IntImm(13), IntImm(33)
        input_0 = Tensor(shape=[N, N], dtype=dtype, name="input_0", is_input=True)

        # Unbound, unfolded constant
        w1_ait = Tensor(shape=[N, N], dtype=dtype, name="w1")

        x1_ait = ops.elementwise(FuncEnum.MUL)(w1_ait, input_0)

        # Unbound folded constant
        w2_ait = Tensor(shape=[N, K], dtype=dtype, name="w2")

        # Bound folded constants
        w3_ait = Tensor(shape=[N, K], dtype=dtype, name="w3")
        w4_ait = Tensor(shape=[N, K], dtype=dtype, name="w4")

        x2_ait = ops.elementwise(FuncEnum.MUL)(w2_ait, w3_ait)
        x3_ait = ops.gemm_rcr()(x2_ait, w4_ait)

        x4_ait = ops.elementwise(FuncEnum.MUL)(x3_ait, x1_ait)

        # Bound unfolded constant
        w5_ait = Tensor(shape=[N, N], dtype=dtype, name="w5")
        output = ops.elementwise(FuncEnum.MUL)(w5_ait, x4_ait)
        output._attrs["is_output"] = True
        output._attrs["name"] = "output"

        input_pt = get_random_torch_tensor((N.value(), N.value()), dtype)
        w1_pt = get_random_torch_tensor((N.value(), N.value()), dtype)
        w2_pt = get_random_torch_tensor((N.value(), K.value()), dtype)
        w3_pt = get_random_torch_tensor((N.value(), K.value()), dtype)
        w4_pt = get_random_torch_tensor((N.value(), K.value()), dtype)
        w5_pt = get_random_torch_tensor((N.value(), N.value()), dtype)

        x1_pt = w1_pt * input_pt
        x2_pt = w2_pt * w3_pt
        x3_pt = torch.nn.functional.linear(x2_pt, w4_pt)
        x4_pt = x3_pt * x1_pt
        output_pt = w5_pt * x4_pt

        mod = compile_model(
            output,
            detect_target(),
            "./tmp",
            "test_constant_folding_mixed_usage",
            constants={"w3": w3_pt, "w4": w4_pt, "w5": w5_pt},
        )

        self.assertSetEqual(
            set(mod.get_constant_folding_input_names()),
            # This is not the only input, but it's the only one we can set.
            {"w2"},
        )

        self.assertSetEqual(set(mod.get_constant_names()), {"w1", "w2"})

        output = torch.empty_like(output_pt)
        # Unset constant W2
        self.assertRaises(RuntimeError, mod.run_with_tensors, [input_pt], [output])
        mod.set_constant_with_tensor("w2", w2_pt)
        # Unset constant W1
        self.assertRaises(RuntimeError, mod.run_with_tensors, [input_pt], [output])
        mod.set_constant_with_tensor("w1", w1_pt)

        mod.run_with_tensors([input_pt], [output])
        torch.testing.assert_close(output, output_pt, atol=1e-1, rtol=1e-1)

    def test_constant_folding_output_in_middle_of_chain(self):
        dtype = "float16"
        N, K = IntImm(13), IntImm(33)
        x = Tensor(shape=[N, K], dtype=dtype, name="x")
        y = Tensor(shape=[N.value() * K.value(), 1], dtype=dtype, name="y")

        x2 = ops.reshape()(x, [N.value() * K.value(), 1])
        x2._attrs["name"] = "x2"
        # Special case: view of constant needed outside of constant folding
        # subgraph.
        x2._attrs["is_output"] = True

        x3 = ops.elementwise(FuncEnum.MUL)(x2, y)
        x3._attrs["name"] = "x3"
        x3._attrs["is_output"] = True

        x4 = ops.elementwise(FuncEnum.ADD)(x3, x3)
        x4._attrs["name"] = "x4"
        x4._attrs["is_output"] = True

        mod = compile_model(
            [x2, x3, x4],
            detect_target(),
            "./tmp",
            "test_constant_folding_output_in_middle_of_chain",
        )

        x_pt = get_random_torch_tensor((N.value(), K.value()), dtype)
        y_pt = get_random_torch_tensor((N.value() * K.value(), 1), dtype)
        x2_pt = x_pt.reshape(N.value() * K.value(), 1)
        x3_pt = x2_pt * y_pt
        x4_pt = x3_pt + x3_pt

        x2_ait, x3_ait, x4_ait = (
            torch.empty_like(x2_pt),
            torch.empty_like(x3_pt),
            torch.empty_like(x4_pt),
        )

        mod.set_many_constants_with_tensors({"x": x_pt, "y": y_pt})
        mod.run_with_tensors([], {"x2": x2_ait, "x3": x3_ait, "x4": x4_ait})

    @parameterized.expand(
        list(
            itertools.product(
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            )
        )
    )
    def test_constant_folding_with_update(
        self,
        update_model_bound: bool = False,
        update_model_unbound: bool = False,
        update_const_folder_bound: bool = False,
        update_const_folder_unbound: bool = False,
        double_buffer: bool = False,
    ):
        input_0 = Tensor(shape=[1, 2], dtype="float16", name="input_0", is_input=True)
        constant_0 = Tensor(shape=[1, 2], dtype="float16", name="constant_0")
        constant_1 = Tensor(shape=[1, 2], dtype="float16", name="constant_1")
        constant_2 = Tensor(shape=[1, 2], dtype="float16", name="constant_2")
        constant_3 = Tensor(shape=[1, 2], dtype="float16", name="constant_3")
        constant_4 = Tensor(shape=[1, 2], dtype="float16", name="constant_4")
        constant_5 = Tensor(shape=[1, 2], dtype="float16", name="constant_5")
        constant_6 = Tensor(shape=[1, 2], dtype="float16", name="constant_6")
        model_constants = {}
        model_unbound_constants = {}
        const_folder_constants = {}
        const_folder_unbound_constants = {}

        # constant 0/1/2 are not folded.
        # constant 0 is unbounded, constant 1/2 is bounded.
        x = ops.elementwise(FuncEnum.MUL)(input_0, constant_0)
        x1 = ops.concatenate()([x, constant_1, constant_2])
        model_constants["constant_1"] = get_random_torch_tensor((1, 2), "float16")
        model_constants["constant_2"] = get_random_torch_tensor((1, 2), "float16")
        model_unbound_constants["constant_0"] = get_random_torch_tensor(
            (1, 2), "float16"
        )

        # constants 3/4/5/6 are folded.
        # constants 3/4 are unbounded, constants 5/6 is bounded.
        y = ops.elementwise(FuncEnum.MUL)(constant_3, constant_4)
        y1 = ops.concatenate()([y, constant_5, constant_6])
        const_folder_unbound_constants["constant_3"] = get_random_torch_tensor(
            (1, 2), "float16"
        )
        const_folder_unbound_constants["constant_4"] = get_random_torch_tensor(
            (1, 2), "float16"
        )
        const_folder_constants["constant_5"] = get_random_torch_tensor(
            (1, 2), "float16"
        )
        const_folder_constants["constant_6"] = get_random_torch_tensor(
            (1, 2), "float16"
        )

        output = ops.elementwise(FuncEnum.MUL)(x1, y1)
        output._attrs["name"] = "output"
        output._attrs["is_output"] = True

        bound_constants = dict(model_constants, **const_folder_constants)
        unbound_constants = dict(
            model_unbound_constants, **const_folder_unbound_constants
        )
        mod = compile_model(
            output,
            detect_target(),
            "./tmp",
            f"test_constant_folding_{update_model_bound}_{update_model_unbound}_{update_const_folder_bound}_{update_const_folder_unbound}_{double_buffer}",
            constants=bound_constants,
        )

        inp0_pt = get_random_torch_tensor((1, 2), "float16")

        def _get_output(new_bound_constants, new_unbound_constants):
            x_pt = inp0_pt * new_unbound_constants["constant_0"]
            x1_pt = torch.cat(
                (
                    x_pt,
                    new_bound_constants["constant_1"],
                    new_bound_constants["constant_2"],
                )
            )
            y = (
                new_unbound_constants["constant_3"]
                * new_unbound_constants["constant_4"]
            )
            y1_pt = torch.cat(
                (
                    y,
                    new_bound_constants["constant_5"],
                    new_bound_constants["constant_6"],
                )
            )
            output_pt = x1_pt * y1_pt
            return output_pt

        output_pt = _get_output(bound_constants, unbound_constants)
        output_ait = torch.empty_like(output_pt)
        mod.set_many_constants_with_tensors(unbound_constants)
        mod.run_with_tensors({"input_0": inp0_pt}, {"output": output_ait})
        self.assertTrue(torch.equal(output_pt, output_ait))

        new_bound_constants = bound_constants
        new_unbound_constants = unbound_constants
        if update_model_bound:
            for k in model_constants.keys():
                new_bound_constants[k] = get_random_torch_tensor((1, 2), "float16")
        if update_model_unbound:
            for k in model_unbound_constants.keys():
                new_unbound_constants[k] = get_random_torch_tensor((1, 2), "float16")

        if update_const_folder_bound:
            for k in const_folder_constants.keys():
                new_bound_constants[k] = get_random_torch_tensor((1, 2), "float16")
        if update_const_folder_unbound:
            for k in const_folder_unbound_constants.keys():
                new_unbound_constants[k] = get_random_torch_tensor((1, 2), "float16")

        if double_buffer:
            mod.set_many_double_buffer_constants_with_tensors(new_bound_constants)
            mod.set_many_double_buffer_constants_with_tensors(new_unbound_constants)
        else:
            mod.set_many_constants_with_tensors(new_bound_constants)
            mod.set_many_constants_with_tensors(new_unbound_constants)
        mod.fold_constants(double_buffer=double_buffer)
        if double_buffer:
            mod.swap_constants()
        mod.run_with_tensors({"input_0": inp0_pt}, {"output": output_ait})
        output_pt = _get_output(new_bound_constants, new_unbound_constants)
        self.assertTrue(torch.equal(output_pt, output_ait))


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
