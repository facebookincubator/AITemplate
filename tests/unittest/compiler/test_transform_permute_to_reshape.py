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
import re
import unittest
from typing import List

import torch

from aitemplate.compiler import compile_model, ops

from aitemplate.compiler.base import IntVar, Tensor
from aitemplate.testing import detect_target, test_utils
from aitemplate.testing.test_utils import get_random_torch_tensor

from parameterized import parameterized

_PERMUTE_OPS = (
    "permute",
    "permute021",
    "permute102",
    "permute210",
    "permute0213",
)


def _generate_model_name(shape, permutation, is_reshape, dtype, is_complex):
    model_name = "_".join(
        [
            ("test_permute_complex" if is_complex else "test_permute"),
            ("to_reshape" if is_reshape else "not_to_reshape"),
            "x".join([str(s) for s in shape]),  #  these  can contain characters
            "".join([str(s) for s in permutation]),  #  unsafe for usage in filenames
            dtype,
        ]
    )
    # replace non-alphanumeric characters with underscores
    # The ^ within the [^a-zA-Z0-9_] is a negation of the
    # character class so it matches every character not in that class,
    model_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
    return model_name


class TransformPermuteToReshapeTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            # no singleton
            ([32, 51, 12], [1, 2, 0], False, False, "float16"),
            ([32, 51, 12], [1, 2, 0], False, False, "float32"),
            # one singleton dimension
            ([32, 51, 1], [0, 2, 1], True, False, "float16"),
            ([32, 51, 1], [0, 2, 1], True, False, "float32"),
            ([32, 51, 1], [1, 2, 0], False, False, "float16"),
            ([32, 51, 1], [0, 2, 1], True, True, "float16"),
            ([32, 51, 1], [1, 2, 0], False, True, "float16"),
            # two same sized dimensions
            ([32, 32, 1], [2, 0, 1], True, False, "float16"),
            ([32, 32, 1], [1, 0, 2], False, False, "float16"),
            # double singleton dimension
            ([32, 1, 51, 1], [3, 0, 2, 1], True, False, "float16"),
            ([32, 1, 51, 1], [2, 3, 1, 0], False, False, "float16"),
            # IntVar dimension
            ([IntVar([1, 10]), 32, 1, 51], [0, 2, 1, 3], True, False, "float16"),
            ([IntVar([1, 10]), 32, 51, 1], [0, 1, 3, 2], True, True, "float16"),
            ([IntVar([1, 10]), 32, 1, 51], [0, 2, 1, 3], True, False, "float32"),
            ([IntVar([1, 10]), 32, 1, 51], [2, 3, 0, 1], False, False, "float16"),
            # other
            ([3, 1, 113, 15, 64], [0, 1, 2, 4, 3], False, False, "float16"),
            ([3, 1, 113, 15, 64], [0, 1, 2, 4, 3], False, False, "float32"),
        ]
    )
    def test_permute_to_reshape(
        self,
        shape: List[int],
        permutation: List[int],
        is_reshape: bool,
        squeeze_trailing_dim: bool,
        dtype: str,
    ):
        target = detect_target()

        if squeeze_trailing_dim:
            # Simulate situation when the rank of the input tensor doesn't
            # match the permutation length, and transform_permute_to_reshape
            # needs to take into account the original shape of the
            # corresponsing tensor accessor. This could happen after fusion of
            # permute and view op by transform_strided_ops pass.
            # We test it by providing an input tensor with last dimension 1 and
            # unsqueezing it before passing to permute
            assert shape[-1] == 1
            X0 = Tensor(shape[:-1], dtype=dtype, is_input=True, name="x")
            X = ops.unsqueeze(len(shape) - 1)(X0)
        else:
            X = Tensor(shape, dtype=dtype, is_input=True, name="x")
        Z = ops.softmax()(ops.permute()(X, dims=permutation), -1)
        Z._attrs["is_output"] = True
        Z._attrs["name"] = "z"

        model_name = _generate_model_name(
            shape, permutation, is_reshape, dtype, is_complex=False
        )
        module = compile_model(Z, target, "./tmp", model_name)
        has_permute_op = any(
            test_utils.graph_has_op(module.debug_sorted_graph, op_name)
            for op_name in _PERMUTE_OPS
        )
        has_reshape_op = test_utils.graph_has_op(module.debug_sorted_graph, "reshape")

        if is_reshape:
            self.assertFalse(has_permute_op)
            self.assertTrue(has_reshape_op)
        else:
            self.assertTrue(has_permute_op)
            self.assertFalse(has_reshape_op)

        shape = [dim.upper_bound() if isinstance(dim, IntVar) else dim for dim in shape]

        x_pt = get_random_torch_tensor(shape, dtype)
        z_pt = torch.softmax(torch.permute(x_pt, tuple(permutation)), dim=-1)
        z_ait = torch.empty_like(z_pt)
        if squeeze_trailing_dim:
            # Same as what we did with AIT input tensor X above
            x_pt = x_pt.squeeze(-1)
        module.run_with_tensors({"x": x_pt}, {"z": z_ait})

        torch.testing.assert_close(z_ait, z_pt, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
