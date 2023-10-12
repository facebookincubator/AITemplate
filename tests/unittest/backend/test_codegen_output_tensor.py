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


import os
import unittest
from typing import Sequence

import torch

from aitemplate.backend.codegen import device_copy, set_value

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target

from aitemplate.testing.test_utils import gen_input_tensor, get_random_torch_tensor


class TestCodegenOutput(unittest.TestCase):
    MODEL_CONTAINER_FILE = "model_container_base.cu"
    MODEL_GENERATED_FILE = "model-generated.h"
    WORKDIR = "/tmp/ait/test_codegen_output"
    SHAPE = (2, 3, 4)

    def _check_string_in_file(self, file_path: str, target_string: str) -> bool:
        with open(file_path, "r") as f:
            content = f.read()
            return target_string in content

    def _assert_codegen_exists(
        self,
        test_name: str,
        expected_codegen: Sequence[str],
        filename: str,
        exists: bool = True,
    ) -> bool:
        model_path = os.path.join(self.WORKDIR, test_name, filename)
        for line in expected_codegen:
            self.assertEqual(
                exists,
                self._check_string_in_file(model_path, line),
                f"Expected this to be codegen'd in {model_path}: \n {line}",
            )

    """
    ### Test Cases:
    1. gelu <-- output
    2. gelu <--view-- output_0
          ^----view-- output_1
    3. gelu <--view-- output_0 <--view-- output_1
    """

    def test_no_view(self, test_name="no_view"):
        """
        Case: Simple case with one output.
        """
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        gelu = ops.elementwise(FuncEnum.GELU)(x)
        output = ops.flatten()(gelu)
        output._attrs["is_output"] = True
        output._attrs["name"] = "output_0"

        compile_model(
            [output],
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=True,
        )

        expected_codegen = (
            # input_x is assigned to params_[0].ptr
            set_value("param_names_[1]", '"output_0"'),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_CONTAINER_FILE
        )

        expected_codegen = (
            set_value(
                "output_0",
                "static_cast<decltype(output_0)>(params_[1].ptr)",
            ),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )

    def test_double_alias(self, test_name="double_alias"):
        """
        Case: Two outputs are a view of the same tensor.
        Graph: ( gelu ) <--view-- ( output_0 )
                        <--view-- ( output_1 )
        Expect: If a tensor is a view for multiple outputs, then it's assigned to
        only one of the outputs' ptrs. We expect D2D copies for the remaining outputs.
        """
        # AIT, two outputs.
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        gelu = ops.elementwise(FuncEnum.GELU)(x)  # has an output alias
        output0 = ops.unsqueeze(dim=0)(gelu)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output_0"

        output1 = ops.flatten()(gelu)
        output1._attrs["is_output"] = True
        output1._attrs["name"] = "output_1"

        model = compile_model(
            [output0, output1],
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=False,
        )

        expected_codegen = (
            set_value("param_names_[1]", '"output_0"'),
            set_value("param_names_[2]", '"output_1"'),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_CONTAINER_FILE
        )

        view = output0._attrs["is_view_of"]
        view_name = view._attrs["name"]
        expected_codegen = (
            set_value("output_0", view_name),
            set_value("output_1", view_name),
            device_copy(output0, view, dst_idx=1),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )

        # This is an edge case -- test the accuracy.
        x_pt = get_random_torch_tensor(self.SHAPE)
        gelu_pt = torch.nn.functional.gelu(x_pt)
        output0_pt = torch.unsqueeze(gelu_pt, dim=0)
        output1_pt = torch.flatten(gelu_pt)
        output0_ait = torch.empty_like(output0_pt)
        output1_ait = torch.empty_like(output1_pt)

        model.run_with_tensors(
            {"input_x": x_pt}, {"output_0": output0_ait, "output_1": output1_ait}
        )
        self.assertTrue(torch.allclose(output0_ait, output0_pt, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(output1_ait, output1_pt, atol=1e-2, rtol=1e-2))

    def test_output_is_view_of_output(self, test_name="output_is_view_of_output"):
        """
        Case: An output is a view of an output.
        Graph: ( gelu ) <--view-- ( output_0 ) <--view-- ( output_1 )
        """
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        gelu = ops.elementwise(FuncEnum.GELU)(x)  # has an output alias
        output0 = ops.flatten()(gelu)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output_0"

        output1 = ops.flatten()(output0)
        output1._attrs["is_output"] = True
        output1._attrs["name"] = "output_1"

        compile_model(
            [output0, output1],
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=False,
        )

        view_name = output0._attrs["is_view_of"]._attrs["name"]
        expected_codegen = (
            set_value("output_0", view_name),
            set_value("output_1", "output_0"),
            device_copy(output1, output0, dst_idx=2),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )
