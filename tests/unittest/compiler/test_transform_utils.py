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

from aitemplate.compiler import compile_model, ops, transform
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target


class TransformUtilsCheckValidityTestCase(unittest.TestCase):
    def _get_simple_graph(self):
        X = Tensor(shape=[2, 3, 4], dtype="float16", name="inputs_0")
        X1 = ops.elementwise(FuncEnum.COS)(X)
        X1._attrs["name"] = "COS"
        X2 = ops.elementwise(FuncEnum.SIN)(X1)
        X2._attrs["name"] = "SIN"
        X3 = ops.elementwise(FuncEnum.ABS)(X1)
        X3._attrs["name"] = "ABS"
        X4 = ops.elementwise(FuncEnum.ADD)(X2, X3)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "ADD"
        return X4

    def test_check_validity_pass(self):
        tensor = self._get_simple_graph()
        graph = transform.toposort(tensor)

        self.assertTrue(
            transform.transform_utils.check_graph_validity(graph),
            "false negative for check_graph_validity",
        )

    def test_check_validity_no_inputs(self):
        tensor = self._get_simple_graph()
        graph = transform.toposort(tensor)

        src_op = list(graph[-2]._attrs["src_ops"])[0]
        src_op._attrs["inputs"] = []
        with self.assertRaisesRegex(
            RuntimeError, "Tensor COS not in inputs for op None"
        ):
            transform.transform_utils.check_graph_validity(graph, raiseError=True)

    def test_check_validity_no_outputs(self):
        tensor = self._get_simple_graph()
        graph = transform.toposort(tensor)

        src_op = list(graph[-1]._attrs["src_ops"])[0]
        src_op._attrs["outputs"] = []
        with self.assertRaisesRegex(
            RuntimeError, "Tensor ADD not in outputs for op None"
        ):
            transform.transform_utils.check_graph_validity(graph, raiseError=True)

    def test_check_validity_no_src_op(self):
        tensor = self._get_simple_graph()
        graph = transform.toposort(tensor)

        graph[-1]._attrs["src_ops"] = set()
        with self.assertRaisesRegex(
            RuntimeError, "Op None not designated as src_op for tensor ADD"
        ):
            transform.transform_utils.check_graph_validity(graph, raiseError=True)

    def test_check_validity_no_dst_op(self):
        tensor = self._get_simple_graph()
        graph = transform.toposort(tensor)

        graph[0]._attrs["dst_ops"] = set()
        with self.assertRaisesRegex(
            RuntimeError, "Op None not designated as dst_op for tensor inputs_0"
        ):
            transform.transform_utils.check_graph_validity(graph, raiseError=True)

    def test_check_validity_lost_input(self):
        tensor = self._get_simple_graph()
        graph = transform.toposort(tensor)

        graph = graph[1:]
        with self.assertRaisesRegex(
            RuntimeError, "Input tensor inputs_0 not established in graph for op None"
        ):
            transform.transform_utils.check_graph_validity(graph, raiseError=True)

    def test_check_validity_lost_output(self):
        tensor = self._get_simple_graph()
        graph = transform.toposort(tensor)

        graph = graph[0:-1]
        with self.assertRaisesRegex(
            RuntimeError, "Output tensor ADD not established in graph for op None"
        ):
            transform.transform_utils.check_graph_validity(graph, raiseError=True)


class TransformUtilsReplaceTensorTestCase(unittest.TestCase):
    def test_original_inputs_replace(self):
        X_shape = [2, 3, 4]
        X = Tensor(shape=X_shape, dtype="float16", name="inputs_0", is_input=True)
        X1 = Tensor(shape=X_shape, dtype="float16", name="inputs_1", is_input=True)

        X2 = ops.elementwise(FuncEnum.COS)(X)
        X3 = ops.elementwise(FuncEnum.SIN)(X1)
        X4 = ops.concatenate()([X2, X3])
        X5 = ops.elementwise(FuncEnum.ADD)(X4, X4)
        X5._attrs["is_output"] = True
        X5._attrs["name"] = "ADD"

        R = ops.elementwise(FuncEnum.COS)(X1)
        transform.transform_utils.remove_dst_op_from_tensor(X1, list(X3.src_ops())[0])
        transform.transform_utils.replace_tensor(X3, R)

        target = detect_target()
        module = compile_model(X5, target, "./tmp", "original_inputs_replace")

        x_pt = torch.randn(X_shape).cuda().half()
        x1_pt = torch.randn(X_shape).cuda().half()
        x2_pt = torch.cos(x_pt)
        r_pt = torch.cos(x1_pt)
        x4_pt = torch.cat([x2_pt, r_pt])
        x5_pt = torch.add(x4_pt, x4_pt)

        y = torch.empty(x5_pt.size()).cuda().half()
        module.run_with_tensors([x_pt, x1_pt], [y])
        self.assertTrue(torch.allclose(x5_pt, y, atol=1e-1, rtol=1e-1))

    def test_is_view_of_replace(self):
        X_shape = [2, 3, 4]
        X = Tensor(shape=X_shape, dtype="float16", name="inputs_0", is_input=True)
        X1 = ops.elementwise(FuncEnum.COS)(X)
        X2 = ops.elementwise(FuncEnum.ABS)(X1)
        X3 = ops.reshape()(X2, [1, 24])
        X4 = ops.elementwise(FuncEnum.SIN)(X3)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "SIN"

        R = ops.elementwise(FuncEnum.COS)(X1)
        transform.transform_utils.remove_dst_op_from_tensor(X1, list(X2.src_ops())[0])
        transform.transform_utils.replace_tensor(X2, R)

        target = detect_target()
        module = compile_model(X4, target, "./tmp", "view_replace")

        x_pt = torch.randn(X_shape).cuda().half()
        x1_pt = torch.cos(x_pt)
        r_pt = torch.cos(x1_pt)
        x3_pt = torch.reshape(r_pt, (1, 24))
        x4_pt = torch.sin(x3_pt)

        y = torch.empty(x4_pt.size()).cuda().half()
        module.run_with_tensors([x_pt], [y])
        self.assertTrue(torch.allclose(x4_pt, y, atol=1e-1, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
