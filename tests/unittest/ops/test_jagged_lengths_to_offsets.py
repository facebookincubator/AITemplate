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
Unittests for the jagged_lengths_to_offsets op.
"""

import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import param, parameterized


class JaggedLengthsToOffsetsTestCase(unittest.TestCase):
    def _test_jagged_lengths_to_offsets(
        self,
        batch_size: int,
        offsets_dtype: str = "int32",
        test_suffix: str = "",
    ):
        LENGTHS = Tensor(
            shape=[IntVar([1, batch_size], name="batch_size")],
            name="lengths",
            dtype=offsets_dtype,
            is_input=True,
        )

        OFFSETS = ops.jagged_lengths_to_offsets()(LENGTHS)

        OFFSETS._attrs["name"] = "offsets"
        OFFSETS._attrs["is_output"] = True

        model = compile_model(
            [OFFSETS],
            detect_target(),
            "./tmp",
            f"test_jagged_lengths_to_offsets_{test_suffix}",
        )

        torch_dtype = string_to_torch_dtype(offsets_dtype)

        for seed in range(10):
            torch.manual_seed(seed)
            lengths_pt = torch.randint(
                low=0,
                high=1024,
                size=(batch_size,),
                dtype=torch_dtype,
            )
            offsets_pt = torch.cat(
                [
                    torch.zeros((1,), dtype=torch_dtype),
                    torch.cumsum(lengths_pt, dim=0, dtype=torch_dtype),
                ],
            ).cuda()

            offsets = torch.empty(
                size=(batch_size + 1,),
                dtype=torch_dtype,
            ).cuda()
            model.run_with_tensors(
                inputs={"lengths": lengths_pt.cuda()},
                outputs=[offsets],
            )

            torch.testing.assert_close(offsets, offsets_pt)

    @parameterized.expand(
        [
            param(1, 1, "int32"),
            param(2, 10, "int64"),
            param(3, 16384, "int32"),
            param(4, 65537, "int64"),
        ]
    )
    def test_jagged_lengths_to_offsets(self, i, batch_size, offsets_dtype):
        self._test_jagged_lengths_to_offsets(
            batch_size=batch_size,
            offsets_dtype=offsets_dtype,
            test_suffix=str(i),
        )


if __name__ == "__main__":
    unittest.main()
