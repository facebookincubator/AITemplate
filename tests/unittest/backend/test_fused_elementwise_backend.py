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
FusedElementwise unittest for backend-agnostic codegen functions.
"""

import unittest

from aitemplate.backend.common.elementwise_common import (
    ElementwiseMetaData,
    FusedElementwiseMetaData,
    gen_function_single_thread,
)
from aitemplate.compiler import ops
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.frontend import Tensor

BATCH_SIZE = 1024
M = 256
K = 1024


class FusedElementwiseCommonCodeGenTestCase(unittest.TestCase):
    def test_unary(self):
        op1 = ops.elementwise(None)
        op2 = ops.elementwise(None)
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=None,
            dst_ops=[op1],
        )
        X2 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op1],
            dst_ops=[op2],
        )
        X3 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op2],
            dst_ops=[],
        )

        fused_func_metadata = FusedElementwiseMetaData(
            inputs=[X1],
            outputs=[X3],
            input_accessors=[TensorAccessor(X1)],
            output_accessors=[TensorAccessor(X3)],
            original_inputs=[X1],
            original_outputs=[X3],
            max_read_t="uint4",
            read_types=["uint4", "uint4"],
            op_t="half2",
            data_t="half",
            input_broadcast_sizes=None,
            dynamic_dims=[],
            sub_funcs=[
                ElementwiseMetaData(
                    func_name="cos",
                    args=[X1],
                    outputs=[X2],
                    op_t="half2",
                ),
                ElementwiseMetaData(
                    func_name="sign",
                    args=[X2],
                    outputs=[X3],
                    op_t="half2",
                ),
            ],
        )

        func_call = gen_function_single_thread(
            fused_func_metadata,
            ["input0"],
            ["output0"],
            None,
        )
        self.assertEqual(func_call, "output0 = sign(cos(input0));\n")

    def test_multi_inputs(self):
        op1 = ops.elementwise(None)
        op2 = ops.elementwise(None)
        op3 = ops.elementwise(None)
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=None,
            dst_ops=[op1],
        )
        X2 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=None,
            dst_ops=[op1],
        )
        X3 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op1],
            dst_ops=[op2],
        )
        X4 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=None,
            dst_ops=[op2],
        )
        X5 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op2],
            dst_ops=[op3],
        )
        X6 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op3],
            dst_ops=[],
        )

        fused_func_metadata = FusedElementwiseMetaData(
            inputs=[X1, X2, X4],
            outputs=[X6],
            input_accessors=[
                TensorAccessor(X1),
                TensorAccessor(X2),
                TensorAccessor(X4),
            ],
            output_accessors=[TensorAccessor(X6)],
            original_inputs=[X1, X2, X4],
            original_outputs=[X6],
            max_read_t="uint4",
            read_types=["uint4", "uint4", "uint4"],
            op_t="half2",
            data_t="half",
            input_broadcast_sizes=None,
            dynamic_dims=[],
            sub_funcs=[
                ElementwiseMetaData(
                    func_name="mul",
                    args=[X1, X2],
                    outputs=[X3],
                    op_t="half2",
                ),
                ElementwiseMetaData(
                    func_name="add",
                    args=[X3, X4],
                    outputs=[X5],
                    op_t="half2",
                ),
                ElementwiseMetaData(
                    func_name="tanh",
                    args=[X5],
                    outputs=[X6],
                    op_t="half2",
                ),
            ],
        )

        func_call = gen_function_single_thread(
            fused_func_metadata,
            ["input0", "input1", "input2"],
            ["output0"],
            None,
        )
        self.assertEqual(func_call, "output0 = tanh(add(mul(input0,input1),input2));\n")

    def test_constant(self):
        op1 = ops.elementwise(None)
        op2 = ops.elementwise(None)
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=None,
            dst_ops=[op1],
        )
        X2 = Tensor(
            shape=[],
            src_ops=None,
            dst_ops=[op1],
            value=10,
        )
        X3 = Tensor(
            shape=[],
            src_ops=None,
            dst_ops=[op2],
            value=20,
        )
        X4 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op1],
            dst_ops=[op2],
        )
        X5 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op2],
            dst_ops=[],
        )

        fused_func_metadata = FusedElementwiseMetaData(
            inputs=[X1],
            outputs=[X5],
            input_accessors=[TensorAccessor(X1)],
            output_accessors=[TensorAccessor(X5)],
            original_inputs=[X1],
            original_outputs=[X5],
            max_read_t="uint4",
            read_types=["uint4"],
            op_t="half2",
            data_t="half",
            input_broadcast_sizes=None,
            dynamic_dims=[],
            sub_funcs=[
                ElementwiseMetaData(
                    func_name="mul",
                    args=[X1, X2],
                    outputs=[X4],
                    op_t="half2",
                ),
                ElementwiseMetaData(
                    func_name="add",
                    args=[X3, X4],
                    outputs=[X5],
                    op_t="half2",
                ),
            ],
        )

        func_call = gen_function_single_thread(
            fused_func_metadata,
            ["input0"],
            ["output0"],
            None,
        )
        self.assertEqual(
            func_call, "output0 = add(half2(20,20),mul(input0,half2(10,10)));\n"
        )

    def test_converter(self):
        op1 = ops.elementwise(None)
        op2 = ops.elementwise(None)
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=None,
            dst_ops=[op1],
        )
        X2 = Tensor(
            shape=[],
            src_ops=None,
            dst_ops=[op1],
            value=10,
        )
        X3 = Tensor(
            shape=[],
            src_ops=None,
            dst_ops=[op2],
            value=20,
        )
        X4 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op1],
            dst_ops=[op2],
        )
        X5 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op2],
            dst_ops=[],
        )
        fused_func_metadata = FusedElementwiseMetaData(
            inputs=[X1],
            outputs=[X5],
            input_accessors=[TensorAccessor(X1)],
            output_accessors=[TensorAccessor(X5)],
            original_inputs=[X1],
            original_outputs=[X5],
            max_read_t="uint4",
            read_types=["uint4"],
            op_t="half",
            data_t="half",
            input_broadcast_sizes=None,
            dynamic_dims=[],
            sub_funcs=[
                ElementwiseMetaData(
                    func_name="mul",
                    args=[X1, X2],
                    outputs=[X4],
                    op_t="float",
                ),
                ElementwiseMetaData(
                    func_name="add",
                    args=[X3, X4],
                    outputs=[X5],
                    op_t="half",
                ),
            ],
        )
        convertors = {
            "half": {"float": "__half2float"},
            "float": {"half": "__float2half_rn"},
        }
        func_call = gen_function_single_thread(
            fused_func_metadata,
            ["input0"],
            ["output0"],
            convertors,
        )
        self.assertEqual(
            func_call,
            "output0 = add(half(20),__float2half_rn(mul(__half2float(input0),float(10))));\n",
        )

    def test_multi_outputs(self):
        op1 = ops.elementwise(None)
        op2 = ops.elementwise(None)
        op3 = ops.elementwise(None)
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=None,
            dst_ops=[op1],
        )
        X2 = Tensor(
            shape=[],
            src_ops=None,
            dst_ops=[op1],
            value=10,
        )
        X3 = Tensor(
            shape=[],
            src_ops=None,
            dst_ops=[op2],
            value=20,
        )
        X4 = Tensor(
            shape=[],
            src_ops=None,
            dst_ops=[op3],
            value=-3,
        )
        X5 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op1],
            dst_ops=[op2, op3],
        )
        X6 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op2],
            dst_ops=[],
        )
        X7 = Tensor(
            shape=[BATCH_SIZE, M, K],
            src_ops=[op3],
            dst_ops=[],
        )

        fused_func_metadata = FusedElementwiseMetaData(
            inputs=[X1],
            outputs=[X6, X7],
            input_accessors=[TensorAccessor(X1)],
            output_accessors=[TensorAccessor(X6), TensorAccessor(X7)],
            original_inputs=[X1],
            original_outputs=[X6, X7],
            max_read_t="uint4",
            read_types=["uint4"],
            op_t="half",
            data_t="half",
            input_broadcast_sizes=None,
            dynamic_dims=[],
            sub_funcs=[
                ElementwiseMetaData(
                    func_name="mul",
                    args=[X1, X2],
                    outputs=[X5],
                    op_t="float",
                ),
                ElementwiseMetaData(
                    func_name="add",
                    args=[X3, X5],
                    outputs=[X6],
                    op_t="half",
                ),
                ElementwiseMetaData(
                    func_name="add",
                    args=[X4, X5],
                    outputs=[X7],
                    op_t="half",
                ),
            ],
        )
        convertors = {
            "half": {"float": "__half2float"},
            "float": {"half": "__float2half_rn"},
        }
        func_call = gen_function_single_thread(
            fused_func_metadata,
            ["input0"],
            ["output0", "output1"],
            convertors,
        )
        self.assertEqual(
            func_call.strip(),
            "\n".join(
                [
                    "half tmp_0 = __float2half_rn(mul(__half2float(input0),float(10)));",
                    "output0 = add(half(20),tmp_0);",
                    "output1 = add(half(-3),tmp_0);",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
