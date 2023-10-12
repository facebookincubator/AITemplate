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

from aitemplate.backend.codegen import set_value

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.testing import detect_target

from aitemplate.testing.test_utils import gen_input_tensor, get_random_torch_tensor


class TestCodegenOutputAliases(unittest.TestCase):
    MODEL_GENERATED_FILE = "model-generated.h"
    WORKDIR = "/tmp/ait/test_codegen_output_aliases"
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
    NOTE: <-- represents a "is_view_of" edge
    1. gelu <-- output
    2. intermediate0 <-- intermediate1 <-- output
    3. gelu <-- output_0
          ^---- output_1
    4. input <-- intermediate <-- output
    5. constant <-- intermediate <-- output
    6. output_0 <-- intermediate <-- output_1
    """

    def test_simple_output_alias(self, test_name="simple_output_alias"):
        """
        Case: Simple case where an output aliases an internal tensor.
        Graph: ( gelu ) <--view-- ( output )
        """
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        gelu = ops.elementwise(FuncEnum.GELU)(x)  # has an output alias
        output0 = ops.flatten()(gelu)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output_0"

        compile_model(
            output0,
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=False,
        )

        view_name = output0._attrs["is_view_of"]._attrs["name"]
        expected_codegen = (
            set_value(
                view_name,
                f"static_cast<decltype({view_name})>(params_[1].ptr)",
            ),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )

    def test_view_of_view(self, test_name="view_of_view"):
        """
        Case: There's two intermediate views, where one is a view of the other.
        Graph: ( input ) <--view-- ( intermediate0 ) <--view-- ( intermediate1 ) <--view-- ( output_0 )
        """
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        intermediate0 = ops.unsqueeze(dim=0)(x)  # has an output alias
        intermediate1 = ops.unsqueeze(dim=0)(intermediate0)  # has an output alias
        output0 = ops.flatten()(intermediate1)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output_0"

        compile_model(
            output0,
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=False,
        )

        intermediate0_name = intermediate0._attrs["name"]
        intermediate1_name = intermediate1._attrs["name"]
        expected_codegen = (
            set_value(intermediate0_name, "input_x"),
            set_value(intermediate1_name, intermediate0_name),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )

    def test_double_alias(self, test_name="double_alias"):
        """
        Case: Two outputs are a view of the same tensor.
        Graph: ( gelu ) <--view-- ( output_0 )
                        <--view-- ( output_1 )
        Expected: gelu will be initialized with params_[2]
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

        compile_model(
            [output0, output1],
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=True,
        )

        view_name = output0._attrs["is_view_of"]._attrs["name"]
        expected_codegen = (
            # input_x assigned to params_[0].ptr
            set_value(
                view_name,
                f"static_cast<decltype({view_name})>(params_[2].ptr)",
                # see _construct_output_name_to_index_map for params indexing
            ),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )

    # ======================================================================== #
    #    Cases where the has_output_alias tensor also has an external tensor   #
    # ======================================================================== #
    def test_external_tensor_is_input(self, test_name="external_tensor_is_input"):
        """
        Case: The intermediate tensor is a view of an input.
        Graph: ( input ) <--view-- ( intermediate ) <--view-- ( output_0 )
        """

        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        intermediate_view = ops.unsqueeze(dim=0)(x)  # has an output alias
        output0 = ops.flatten()(intermediate_view)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output_0"

        compile_model(
            output0,
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=False,
        )
        intermediate_view_name = intermediate_view._attrs["name"]
        expected_codegen = (
            # input_x is assigned to params_[0].ptr
            set_value(intermediate_view_name, "input_x"),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )

    def test_external_tensor_is_constant(self, test_name="external_tensor_is_constant"):
        """
        Case: The intermediate tensor is a view of a constant.
        Graph: ( constant ) <--view-- ( intermediate ) <--view-- ( output_0 )
        Expect: The intermediate tensor is constant folded.
        """
        constant = Tensor(
            shape=self.SHAPE, dtype="float16", name="constant"
        )  # has an output alias
        intermediate_view = ops.unsqueeze(dim=0)(constant)  # has an output alias
        output0 = ops.flatten()(intermediate_view)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output_0"

        constant_pt = get_random_torch_tensor(self.SHAPE, dtype="float16")

        compile_model(
            output0,
            detect_target(),
            self.WORKDIR,
            test_name,
            constants={"constant": constant_pt},
            do_optimize_graph=False,
        )
        intermediate_view_name = intermediate_view._attrs["name"]
        expected_codegen = (
            set_value(intermediate_view_name, "constant"),
            set_value("output_0", intermediate_view_name),
        )
        # We expect the intermediate tensor and output_0 to be constant folded.
        # So, we don't expect _codegen_output_aliases_tensor to be called.
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE, exists=False
        )

    def test_external_tensor_is_output(self, test_name="external_tensor_is_output"):
        """
        Case: The intermediate tensor is a view of an output.
        Graph: ( output_0 ) <--view-- ( intermediate ) <--view-- ( output_1 )
        """
        x = gen_input_tensor(shape=self.SHAPE, name="input_x")
        z = gen_input_tensor(shape=self.SHAPE, name="input_z")
        output0 = ops.elementwise(FuncEnum.RELU)(x + z)
        output0._attrs["is_output"] = True
        output0._attrs["name"] = "output_0"

        intermediate_view = ops.unsqueeze(dim=0)(output0)  # has an output alias
        output1 = ops.flatten()(intermediate_view)
        output1._attrs["is_output"] = True
        output1._attrs["name"] = "output_1"

        compile_model(
            [output0, output1],
            detect_target(),
            self.WORKDIR,
            test_name,
            do_optimize_graph=False,
        )
        intermediate_name = intermediate_view._attrs["name"]
        expected_codegen = (
            # input_z is assigned to params_[0].ptr
            # input_x is assigned to params_[1].ptr
            set_value(
                "output_0",
                "static_cast<decltype(output_0)>(params_[2].ptr)",
            ),
            set_value(intermediate_name, "output_0"),
        )
        self._assert_codegen_exists(
            test_name, expected_codegen, self.MODEL_GENERATED_FILE
        )
