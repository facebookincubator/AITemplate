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
import logging
import os
import re
import subprocess
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils.debug_settings import AITDebugSettings
from aitemplate.utils.misc import is_windows

_LOGGER = logging.getLogger(__name__)


class StridedOpCatPatternTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_gen_standalone(self, test_name, dtype):
        M = 8
        N = 16
        K = 32
        X1 = Tensor(
            shape=[IntImm(M), IntImm(K)],
            dtype=dtype,
            name="X1",
            is_input=True,
        )
        W1 = Tensor(
            shape=[IntImm(N), IntImm(K)],
            dtype=dtype,
            name="W1",
            is_input=True,
        )
        B1 = Tensor(
            shape=[IntImm(N)],
            dtype=dtype,
            name="B1",
            is_input=True,
        )
        X2 = Tensor(
            shape=[IntImm(M), IntImm(N)],
            dtype=dtype,
            name="X2",
            is_input=True,
        )
        X3 = Tensor(
            shape=[IntImm(M), IntImm(N)],
            dtype=dtype,
            name="X3",
            is_input=True,
        )
        Y1 = ops.gemm_rcr_bias()(X1, W1, B1)
        Y2 = ops.elementwise(FuncEnum.ADD)(Y1, X2)
        cat_dim = 1
        Y = ops.concatenate()([X3, Y2], dim=cat_dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        target = detect_target()
        debug_settings = AITDebugSettings(gen_standalone=True)
        dll_name = "test.so"
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            dll_name=dll_name,
            debug_settings=debug_settings,
        )

        x1_pt = get_random_torch_tensor([M, K], dtype)
        w1_pt = get_random_torch_tensor([N, K], dtype)
        b1_pt = get_random_torch_tensor([N], dtype)
        x2_pt = get_random_torch_tensor([M, N], dtype)
        x3_pt = get_random_torch_tensor([M, N], dtype)

        y1_pt = torch.nn.functional.linear(x1_pt, w1_pt, b1_pt)
        y2_pt = y1_pt + x2_pt
        y_pt = torch.cat([x3_pt, y2_pt], dim=cat_dim)
        y = get_torch_empty_tensor(y_pt.shape, dtype)

        module.run_with_tensors(
            {
                "X1": x1_pt,
                "W1": w1_pt,
                "B1": b1_pt,
                "X2": x2_pt,
                "X3": x3_pt,
            },
            [y],
        )
        self.assertTrue(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))

        # Now we run the generated executable
        cwd = os.getcwd()
        workdir = os.path.join(cwd, "tmp", test_name)
        working_env = os.environ.copy()
        if "LD_LIBRARY_PATH" in working_env:
            working_env["LD_LIBRARY_PATH"] = (
                working_env["LD_LIBRARY_PATH"] + ":" + workdir
            )
        else:
            working_env["LD_LIBRARY_PATH"] = workdir
        _LOGGER.info(f"work dir: {workdir}")
        exe_name = "./test.exe" if is_windows() else "./test"
        with subprocess.Popen(
            [exe_name],
            shell=True,
            cwd=workdir,
            env=working_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            try:
                timeout = 10
                out, err = proc.communicate(timeout)
            except subprocess.TimeoutExpired as e:
                proc.kill()
                out, err = proc.communicate()
                raise e
            finally:
                stdout = out.decode()
                stderr = err.decode()
                if proc.returncode != 0:
                    _LOGGER.info(f"stdout:\n\n{stdout}")
                    _LOGGER.info(f"stderr:\n\n{stderr}")
                    raise RuntimeError(f"failed to execute {exe_name}")
                else:
                    _LOGGER.info(f"stdout:\n\n{stdout}")
                    all_output_lines = stdout.split("\n")
                    output_lines = [
                        line for line in all_output_lines if "output_0" in line
                    ]
                    self.assertTrue(len(output_lines) == 1)
                    m = re.search("with shape: +([0-9,]+)", output_lines[0])
                    self.assertTrue(m is not None)
                    shape = m.group(1).split(",")
                    self.assertTrue(int(shape[0]) == 8)
                    self.assertTrue(int(shape[1]) == 32)

    def test_gen_standalone_f16(self):
        self._test_gen_standalone("gen_standalone_f16", "float16")

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_gen_standalone_f32_sm80(self):
        self._test_gen_standalone("gen_standalone_f32", "float32")


filter_test_cases_by_test_env(StridedOpCatPatternTestCase)

if __name__ == "__main__":
    unittest.main()
