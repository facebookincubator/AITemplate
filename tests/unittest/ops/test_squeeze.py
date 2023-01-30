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
from typing import List, Optional, Tuple

import torch
from aitemplate.compiler import compile_model

from aitemplate.compiler.ops import elementwise, squeeze, unsqueeze
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


def _construct_shape(
    shape: List[List[int]], input_number: int
) -> Tuple[List[IntVar], List[Optional[str]]]:
    result = []
    dim_names = []
    num_dynamic = 0
    for dim in shape:
        dim_name = None
        if len(dim) == 1:
            result.append(IntImm(dim[0]))
        else:
            dim_name = f"dynamic{input_number}{num_dynamic}"
            result.append(IntVar(dim, name=dim_name))
            num_dynamic += 1
        dim_names.append(dim_name)
    return result, dim_names


class SqueezeTestCase(unittest.TestCase):
    def _test_helper(
        self, dim, shape, expected_shape, test_name, do_squeeze, input_type="float16"
    ):
        target = detect_target()

        shape_vars, input_0_names = _construct_shape(shape, 0)
        expected_shape_vars, input_1_names = _construct_shape(expected_shape, 0)

        input_0 = Tensor(
            shape=shape_vars, dtype=input_type, name="input_0", is_input=True
        )
        input_1 = Tensor(
            shape=expected_shape_vars, dtype=input_type, name="input_1", is_input=True
        )

        if do_squeeze:
            op = squeeze(dim)
        else:
            op = unsqueeze(dim)

        Y = op(input_0)
        # Elementwise multiply with another input tensor with the expected shape
        # This makes sure that squeeze/unsqueeze infer the correct shape
        output = elementwise(FuncEnum.MUL)(Y, input_1)

        output._attrs["name"] = "output_0"
        output._attrs["is_output"] = True

        module = compile_model(output, target, "./tmp", test_name)

        all_input_0_shapes = itertools.product(*shape)
        all_input_1_shapes = itertools.product(*expected_shape)

        for input_0_shape, input_1_shape in zip(all_input_0_shapes, all_input_1_shapes):
            input_0_pt = get_random_torch_tensor(input_0_shape, input_type)
            input_1_pt = get_random_torch_tensor(input_1_shape, input_type)
            if do_squeeze:
                # For some reason, torch.squeeze(X_pt, dim) fails when
                # dim is None (even though the docs say dim is Optional[int])!
                if dim is not None:
                    Y_pt = torch.squeeze(input_0_pt, dim=dim)
                else:
                    Y_pt = torch.squeeze(input_0_pt)
            else:
                Y_pt = torch.unsqueeze(input_0_pt, dim)

            output_pt = torch.mul(Y_pt, input_1_pt)
            inputs = [input_0_pt, input_1_pt]

            output = torch.empty_like(input_1_pt)
            module.run_with_tensors(inputs, [output])
            # outputs from view ops must be exactly equal
            self.assertTrue(torch.equal(output, output_pt))

    def test_squeeze(self):
        self._test_helper(
            None, [[4, 3], [1], [2], [1]], [[4, 3], [2]], "squeeze0", True
        )
        self._test_helper(0, [[1], [1], [2], [1]], [[1], [2], [1]], "squeeze1", True)
        self._test_helper(
            2, [[4, 2], [4], [1], [8]], [[4, 2], [4], [8]], "squeeze2", True
        )
        self._test_helper(-2, [[6], [1], [1], [16]], [[6], [1], [16]], "squeeze3", True)

    def test_unsqueeze(self):
        self._test_helper(
            1,
            [[4, 3], [1], [2], [1]],
            [[4, 3], [1], [1], [2], [1]],
            "unsqueeze0",
            False,
        )
        self._test_helper(
            0,
            [[4, 3], [1], [2], [1]],
            [[1], [4, 3], [1], [2], [1]],
            "unsqueeze1",
            False,
        )
        self._test_helper(
            -1,
            [[4, 3], [1], [2], [3]],
            [[4, 3], [1], [2], [3], [1]],
            "unsqueeze2",
            False,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_squeeze_float32(self):
        self._test_helper(
            2,
            [[4, 2], [4], [1], [8]],
            [[4, 2], [4], [8]],
            "squeeze_float32",
            True,
            input_type="float32",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_unsqueeze_float32(self):
        self._test_helper(
            0,
            [[4, 3], [1], [2], [1]],
            [[1], [4, 3], [1], [2], [1]],
            "unsqueeze_float32",
            False,
            input_type="float32",
        )


if __name__ == "__main__":
    unittest.main()
