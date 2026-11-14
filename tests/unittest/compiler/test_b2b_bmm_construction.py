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
from unittest.mock import patch

from aitemplate.compiler import ops
from aitemplate.compiler.ops.b2b_bmm.b2b_bmm_base import CausalType


class B2bBmmConstructionTestCase(unittest.TestCase):
    def test_construct_without_cutlass(self):
        attributes = {
            "causal_type": CausalType.NO_CAUSAL,
            "epilogue_math_name": "Identity",
            "alpha0": 0.5,
            "alpha1": 1.0,
            "alpha1_divide_by_seq_len": True,
        }
        with patch.dict("sys.modules", {"cutlass_lib": None}):
            for op_type in (
                ops.classic_b2b_bmm,
                ops.fmha_style_b2b_bmm,
                ops.grouped_classic_b2b_bmm,
                ops.grouped_fmha_style_b2b_bmm,
            ):
                with self.subTest(op_type=op_type):
                    op = op_type(**attributes)
                    copied_op = op_type(**op._get_op_attributes())
                    self.assertEqual(op._get_op_attributes(), attributes)
                    self.assertEqual(copied_op._get_op_attributes(), attributes)


if __name__ == "__main__":
    unittest.main()
