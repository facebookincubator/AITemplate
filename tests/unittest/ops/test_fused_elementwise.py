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
"""
Unittests for fused_elementwise Operator.
"""

import itertools
import unittest
from typing import List

import torch

from aitemplate.compiler import compile_model, ops, transform
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.stable_set import StableSet
from aitemplate.compiler.transform.fuse_ops import _get_inputs_outputs
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
    get_torch_full_tensor,
)
from aitemplate.utils import shape_utils
from parameterized import parameterized

ait_dtype_to_pytorch = {"float16": torch.float16}
if detect_target().name() != "rocm":
    ait_dtype_to_pytorch["float32"] = torch.float32
    if int(detect_target()._arch) >= 80:
        ait_dtype_to_pytorch["bfloat16"] = torch.bfloat16


class FusedElementwiseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_fused_elementwise_constructor(self, ait_dtype):
        BATCH_SIZE = 1024
        M = 256
        K = 128

        op1 = ops.elementwise(FuncEnum.ADD)
        op1._attrs["name"] = "e1"
        op2 = ops.elementwise(FuncEnum.TANH)
        op2._attrs["name"] = "e2"
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype=ait_dtype,
            name="X2",
            value=3.0,
        )
        X3 = op1(X1, X2)
        X3._attrs["name"] = "X3"
        X4 = op2(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True

        graph = transform.toposort(X4)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        transform.refine_graph(graph)
        inputs, outputs, external_inputs, external_outputs = _get_inputs_outputs(
            {op1, op2}, {op1, op2}
        )
        for tensor in inputs | outputs:
            tensor._attrs["src_ops"] = tensor._attrs["src_ops"] - {op1, op2}
            tensor._attrs["dst_ops"] = tensor._attrs["dst_ops"] - {op1, op2}
        fused_op = ops.fused_elementwise(
            [op1, op2],
            external_inputs,
            external_outputs,
        )
        fused_op._attrs["name"] = "fused_elementwise0"

        self.assertEqual(fused_op._attrs["inputs"], [X1])
        self.assertEqual(fused_op._attrs["outputs"], [X4])

        self.assertEqual(X4._attrs["src_ops"], StableSet({fused_op}))
        self.assertEqual(X1._attrs["dst_ops"], StableSet({fused_op}))

        self.assertEqual(fused_op._attrs["depth"], 0)
        self.assertEqual(X1._attrs["depth"], 0)
        self.assertEqual(X4._attrs["depth"], 2)

    def test_fused_elementwise_constructor(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_fused_elementwise_constructor(ait_dtype)

    def _test_fused_elementwise_e2e(self, batch_sizes, ms, ks, test_name, ait_dtype):
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X1 = Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(batch_sizes),
                shape_utils.gen_int_var_min_max(ms),
                shape_utils.gen_int_var_min_max(ks),
            ],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype=ait_dtype,
            name="X2",
            value=3.0,
        )
        X3 = X1 + X2
        X3._attrs["name"] = "X3"
        X4 = ops.tanh(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(
            X4,
            target,
            "./tmp",
            "fused_elementwise_{}".format(test_name),
        )

        for batch_size in batch_sizes:
            for m in ms:
                for k in ks:
                    x1_pt = torch.randn(batch_size, m, k).cuda().to(dtype=torch_dtype)
                    x4_pt = torch.tanh(x1_pt + 3.0)

                    x4 = torch.empty([batch_size, m, k]).cuda().to(dtype=torch_dtype)
                    module.run_with_tensors([x1_pt], [x4])
                    self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-2, rtol=1e-2))

    def test_fused_elementwise_e2e(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_fused_elementwise_e2e(
                batch_sizes=[1024],
                ms=[256],
                ks=[128],
                test_name=f"static_shapes_{ait_dtype}",
                ait_dtype=ait_dtype,
            )
            self._test_fused_elementwise_e2e(
                batch_sizes=[1, 99, 998, 1024],
                ms=[256],
                ks=[128],
                test_name=f"dynamic_batch_size_{ait_dtype}",
                ait_dtype=ait_dtype,
            )
            self._test_fused_elementwise_e2e(
                batch_sizes=[1024],
                ms=[1, 128, 256],
                ks=[128],
                test_name=f"dynamic_m_{ait_dtype}",
                ait_dtype=ait_dtype,
            )
            self._test_fused_elementwise_e2e(
                batch_sizes=[1024],
                ms=[256],
                ks=[1, 3, 8, 128],
                test_name=f"dynamic_k_{ait_dtype}",
                ait_dtype=ait_dtype,
            )
            self._test_fused_elementwise_e2e(
                batch_sizes=[700, 80, 1024],
                ms=[23, 78, 256],
                ks=[10, 30, 128],
                test_name=f"dynamic_all_{ait_dtype}",
                ait_dtype=ait_dtype,
            )

    def _test_fused_elementwise_kernel1(self, ait_dtype):
        BATCH_SIZE = 1024
        M = 1496
        X1 = Tensor(
            shape=[IntImm(BATCH_SIZE), IntImm(2), IntImm(M)],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype=ait_dtype,
            name="constant_number",
            value=2.0,
        )
        X3 = Tensor(
            shape=[IntImm(2), IntImm(M)],
            dtype=ait_dtype,
            name="constant_matrix",
            is_input=True,
        )
        X4 = ops.elementwise(FuncEnum.SIGN)(X1)
        X5 = ops.elementwise(FuncEnum.ABS)(X1)
        X6 = ops.elementwise(FuncEnum.ADD)(X5, X2)
        X7 = ops.elementwise(FuncEnum.LOGE)(X6)
        X8 = ops.elementwise(FuncEnum.MUL)(X4, X7)
        X9 = ops.elementwise(FuncEnum.MUL)(X8, X3)
        X9._attrs["is_output"] = True
        X9._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(
            X9, target, "./tmp", f"fused_elementwise_kernel1_{ait_dtype}"
        )

        x1_pt = get_random_torch_tensor((BATCH_SIZE, 2, M), ait_dtype)
        x3_pt = get_random_torch_tensor((2, M), ait_dtype)
        x9_pt = torch.sign(x1_pt) * torch.log1p(torch.abs(x1_pt) + 1) * x3_pt

        inputs = {"input0": x1_pt, "constant_matrix": x3_pt}
        x9 = get_torch_empty_tensor([BATCH_SIZE, 2, M], ait_dtype)
        module.run_with_tensors(inputs, [x9])
        torch.testing.assert_close(x9, x9_pt, atol=1e-2, rtol=1e-2)

    def test_fused_elementwise_kernel1(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_fused_elementwise_kernel1(ait_dtype)

    def _test_sigmoid(self, input_size, test_name, ait_dtype):
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.SIGMOID)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = (
            (torch.rand(input_size, device="cuda", dtype=torch_dtype) - 0.5) * 2.0
        ) * torch.finfo(torch_dtype).max
        x2_pt = torch.sigmoid(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))
        # sanity checks
        self.assertEqual(torch.sum(x2 < 0), 0)
        self.assertEqual(torch.sum(x2 > 1), 0)

    def test_sigmoid(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_sigmoid([1024, 2 * 1496], f"sigmoid_1_{ait_dtype}", ait_dtype)
            self._test_sigmoid([1024, 23744], f"sigmoid_2_{ait_dtype}", ait_dtype)
            self._test_sigmoid([1024, 70144], f"sigmoid_3_{ait_dtype}", ait_dtype)

    def _test_tanh(self, input_size, test_name, ait_dtype):
        assert len(input_size) == 2
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.TANH)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.tanh(x1_pt)

        x2 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def test_tanh(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_tanh([1024, 22400], f"tanh_1_{ait_dtype}", ait_dtype)
            self._test_tanh([1024, 70144], f"tanh_2_{ait_dtype}", ait_dtype)
            self._test_tanh([1024, 23744], f"tanh_3_{ait_dtype}", ait_dtype)

    def _test_gelu(self, input_size, test_name, ait_dtype, fast_gelu=False):
        assert len(input_size) == 2
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        if fast_gelu:
            X2 = ops.elementwise(FuncEnum.FASTGELU)(X1)
        else:
            X2 = ops.elementwise(FuncEnum.GELU)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.nn.functional.gelu(x1_pt)

        x2 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    def test_gelu(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_gelu([1024, 22400], f"gelu_1_{ait_dtype}", ait_dtype)
            self._test_gelu([1024, 70144], f"fast_gelu_1_{ait_dtype}", ait_dtype, True)

    def _test_power(self, input_size, exp, test_name, ait_dtype):
        print(f"Running test {test_name} with exp = {exp}")
        assert len(input_size) == 2
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.POW)(X1, exp)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        if abs(exp) < 1.0:
            x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype) + 0.5
        else:
            x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.pow(x1_pt, exp)

        x2 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [x2])
        # t, _, _ = module.benchmark_with_tensors([x1_pt], [x2], count=1000)
        # bw = input_size[0] * input_size[1] * 2 * 2 / (t * 1e9 * 1e-3)
        # print(f"BW: {bw} GB/s")
        torch.testing.assert_close(x2, x2_pt, atol=1e-3, rtol=1e-3, equal_nan=True)

    @parameterized.expand(
        itertools.product(
            (0, 1, -1, 0.5, -0.5, 2, -2, 1.4, 3),
            ([1024, 1024], [1025, 1025]),
        )
    )
    def test_power_float16(self, exp, shape):
        dtype = "float16"
        self._test_power(
            shape,
            exp,
            f"pow_{shape[0]}_{shape[1]}_{exp}_{dtype}",
            dtype,
        )

    @unittest.skipIf(
        detect_target().name() != "cuda", "float32 dtype only supported in CUDA"
    )
    def test_power_float32(self):
        self._test_power(
            (1024, 1024),
            2.5,
            "pow_float32",
            "float32",
        )

    @unittest.skipIf(
        detect_target().name() != "cuda", "bfloat16 dtype only supported in CUDA"
    )
    @unittest.skipIf(
        int(detect_target()._arch) < 80, "bfloat16 dtype only supported in CUDA sm80+"
    )
    def test_power_bfloat16(self):
        self._test_power(
            (1024, 1024),
            1.2,
            "pow_bfloat16",
            "bfloat16",
        )

    def _test_min_max(
        self,
        input_size: List[List[int]],
        test_name: str,
        is_min: bool,
        add_nans: bool,
        ait_dtype,
    ) -> None:
        assert len(input_size) == 2
        X0 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=ait_dtype,
            name="input1",
            is_input=True,
        )
        if is_min:
            result = ops.elementwise(FuncEnum.MIN)(X0, X1)
        else:
            result = ops.elementwise(FuncEnum.MAX)(X0, X1)

        result._attrs["is_output"] = True
        result._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(result, target, "./tmp", test_name)

        x0_pt = get_random_torch_tensor(input_size, ait_dtype)
        x1_pt = get_random_torch_tensor(input_size, ait_dtype)
        if add_nans:
            x1_pt[0].fill_(float("nan"))

        if is_min:
            x2_pt = torch.min(x0_pt, x1_pt)
        else:
            x2_pt = torch.max(x0_pt, x1_pt)

        inputs = {"input0": x0_pt, "input1": x1_pt}
        x2 = get_torch_empty_tensor(input_size, ait_dtype)
        module.run_with_tensors(inputs, [x2])

        if add_nans:
            nans = get_torch_full_tensor(x2_pt[0].shape, float("nan"), ait_dtype)
            torch.testing.assert_close(nans, x2_pt[0], equal_nan=True)
            torch.testing.assert_close(nans, x2[0], equal_nan=True)

        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2, equal_nan=True)

    def test_min(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_min_max(
                [512, 512],
                test_name=f"min_nonan_{ait_dtype}",
                is_min=True,
                add_nans=False,
                ait_dtype=ait_dtype,
            )
            self._test_min_max(
                [512, 512],
                test_name=f"min_nan_{ait_dtype}",
                is_min=True,
                add_nans=True,
                ait_dtype=ait_dtype,
            )

    def test_max(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_min_max(
                [512, 512],
                test_name=f"max_nonan_{ait_dtype}",
                is_min=False,
                add_nans=False,
                ait_dtype=ait_dtype,
            )
            self._test_min_max(
                [512, 512],
                test_name=f"max_nan_{ait_dtype}",
                is_min=False,
                add_nans=True,
                ait_dtype=ait_dtype,
            )

    def _test_clamp(
        self,
        input_size: List[List[int]],
        min_val: int,
        max_val: int,
        test_name: str,
        ait_dtype,
    ) -> None:
        assert len(input_size) == 2 or len(input_size) == 0
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X0 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])] if input_size else [],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        result = ops.clamp()(X0, min_val, max_val)
        result._attrs["is_output"] = True
        result._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(result, target, "./tmp", test_name)

        x0_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)

        x1_pt = torch.clamp(x0_pt, min_val, max_val)

        x1 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x0_pt], [x1])

        self.assertTrue(torch.allclose(x1, x1_pt, atol=1e-2, rtol=1e-2))

    def test_clamp(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_clamp([512, 106], -1, 1, f"clamp_0_{ait_dtype}", ait_dtype)
            self._test_clamp([128, 46], None, 1, f"clamp_1_{ait_dtype}", ait_dtype)
            self._test_clamp([56, 265], -1, None, f"clamp_2_{ait_dtype}", ait_dtype)
            self._test_clamp([17, 123], 1, -1, f"clamp_3_{ait_dtype}", ait_dtype)
            self._test_clamp([], 1, -1, f"clamp_4_{ait_dtype}", ait_dtype)

    def _test_operator_overload(self, ait_dtype):
        input_size = [4, 2]
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X1 = Tensor(
            shape=input_size,
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=input_size,
            dtype=ait_dtype,
            name="input1",
            is_input=True,
        )
        OUTPUT = -ops.tanh(X1 + X2) + ops.tanh(X2) + ops.tanh(X1)
        OUTPUT._attrs["is_output"] = True
        OUTPUT._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(OUTPUT, target, "./tmp", f"test_op_overload_{ait_dtype}")

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        output_pt = -torch.tanh(x1_pt + x2_pt) + torch.tanh(x2_pt) + torch.tanh(x1_pt)

        output = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt, x2_pt], [output])
        self.assertTrue(torch.allclose(output, output_pt, atol=1e-2, rtol=1e-2))

    def test_operator_overload(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_operator_overload(ait_dtype)

    def _test_operator_overload_with_constant_number(self, ait_dtype):
        input_size = [4, 2]
        torch_dtype = ait_dtype_to_pytorch[ait_dtype]
        X1 = Tensor(
            shape=input_size,
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        OUTPUT = 10 / ops.tanh(X1 + 5)
        OUTPUT._attrs["is_output"] = True
        OUTPUT._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(OUTPUT, target, "./tmp", f"test_op_overload_{ait_dtype}")

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        output_pt = 10 / torch.tanh(x1_pt + 5)
        output = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [output])
        self.assertTrue(torch.allclose(output, output_pt, atol=1e-2, rtol=1e-2))

    def test_operator_overload_with_constant_number(self):
        for ait_dtype in ait_dtype_to_pytorch.keys():
            self._test_operator_overload_with_constant_number(ait_dtype)


if __name__ == "__main__":
    unittest.main()
