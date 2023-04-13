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
from unittest.mock import patch

import jinja2
from aitemplate.backend.build_cache_base import SkipBuildCache

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import DynamicProfileStrategy
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target


class _EnableForceProfile:
    """
    Prevent cached profiling entries from causing profiling-related
    compilation tests from failing.
    """

    def __init__(self):
        self.old_force_profile = os.environ.get("FORCE_PROFILE", None)

    def __enter__(self):
        os.environ["FORCE_PROFILE"] = "1"
        return self

    def __exit__(self, *args):
        if self.old_force_profile is None:
            del os.environ["FORCE_PROFILE"]
        else:
            os.environ["FORCE_PROFILE"] = self.old_force_profile


class CompilationFailureTestCase(unittest.TestCase):
    def _test_compilation_failure(
        self,
        test_name="compilation_failure",
    ):
        target = detect_target()

        X = Tensor(
            shape=[IntImm(4), 28, 28, 128],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 3, 3, 128],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        OP = ops.conv2d(stride=1, pad=1, dilate=1)
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        with SkipBuildCache():
            compile_model(
                Y,
                target,
                f"./tmp/{test_name}",
                test_name,
                dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
            )

    def test_compilation_failure_profiler(self):
        target = detect_target().name()
        with _EnableForceProfile():
            profiler_main_template = (
                f"aitemplate.backend.{target}.conv2d.common.PROFILER_MAIN_TEMPLATE"
            )
            with patch(profiler_main_template, jinja2.Template("BAD CODE!")):
                with self.assertRaisesRegex(RuntimeError, "Build has failed."):
                    self._test_compilation_failure(
                        test_name="compilation_failure_profiler"
                    )

    def test_compilation_failure_function(self):
        target = detect_target().name()
        gen_function = f"aitemplate.backend.{target}.conv2d.common.gen_function"
        with patch(gen_function) as mock_gen_function:
            mock_gen_function.return_value = "BAD CODE!"
            with self.assertRaisesRegex(RuntimeError, "Build has failed."):
                self._test_compilation_failure(test_name="compilation_failure_function")


if __name__ == "__main__":
    unittest.main()
