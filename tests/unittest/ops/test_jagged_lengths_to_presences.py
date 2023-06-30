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
Unittests for the jagged_lengths_to_presences op.
"""

import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.torch_utils import string_to_torch_dtype
from parameterized import param, parameterized


def _compute_presences_pt(
    lengths_pt: torch.Tensor,
    max_seq_len: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    data = []
    for length in lengths_pt.cpu().tolist():
        data.append([1] * length + [0] * (max_seq_len - length))
    return torch.tensor(data, dtype=output_dtype)


class JaggedLengthsToPresencesTestCase(unittest.TestCase):
    def _test_jagged_lengths_to_presences(
        self,
        batch_size: int,
        max_seq_len: int = 128,
        lengths_dtype: str = "int32",
        presences_dtype: str = "float16",
        test_suffix: str = "",
    ):
        LENGTHS = Tensor(
            shape=[IntVar([1, batch_size], name="batch_size")],
            name="lengths",
            dtype=lengths_dtype,
            is_input=True,
        )

        PRESENCES = ops.jagged_lengths_to_presences()(
            lengths=LENGTHS,
            max_seq_len=max_seq_len,
            dtype=presences_dtype,
        )

        PRESENCES._attrs["name"] = "presences"
        PRESENCES._attrs["is_output"] = True

        model = compile_model(
            [PRESENCES],
            detect_target(),
            "./tmp",
            f"test_jagged_lengths_to_presences_{test_suffix}",
        )

        torch_lengths_dtype = string_to_torch_dtype(lengths_dtype)
        torch_presences_dtype = string_to_torch_dtype(presences_dtype)

        for seed in range(10):
            torch.manual_seed(seed)
            lengths_pt = torch.randint(
                low=0,
                high=max_seq_len,
                size=(batch_size,),
                dtype=torch_lengths_dtype,
            ).cuda()
            presences_pt = _compute_presences_pt(
                lengths_pt=lengths_pt,
                max_seq_len=max_seq_len,
                output_dtype=torch_presences_dtype,
            ).cuda()

            presences = torch.empty(
                size=(batch_size, max_seq_len),
                dtype=torch_presences_dtype,
            ).cuda()
            model.run_with_tensors(
                inputs={"lengths": lengths_pt},
                outputs=[presences],
            )

            torch.testing.assert_close(presences, presences_pt)

    @parameterized.expand(
        [
            param(1, 1, 1, "int32", "bool"),
            param(2, 11, 23, "int64", "float32"),
            param(3, 1024, 256, "int32", "float16"),
            param(4, 1234, 567, "int64", "bool"),
        ]
    )
    def test_jagged_lengths_to_presences(
        self,
        i,
        batch_size,
        max_seq_len,
        lengths_dtype,
        presences_dtype,
    ):
        self._test_jagged_lengths_to_presences(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            lengths_dtype=lengths_dtype,
            presences_dtype=presences_dtype,
            test_suffix=str(i),
        )


if __name__ == "__main__":
    unittest.main()
