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

import sympy

from aitemplate.compiler import ops, symbolic
from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum


class SymbolTestCase(unittest.TestCase):
    def test_symbolic_values_existence(self):
        imm = IntImm(value=7)
        self.assertIsNotNone(imm._attrs.get("symbolic_value", None))

        var = IntVar(values=[1, 256])
        self.assertIsNotNone(var._attrs.get("symbolic_value", None))

    def test_imm_equal(self):
        imm1 = IntImm(value=7)
        imm2 = IntImm(value=8)
        imm3 = IntImm(value=7, name="dummy_name")

        self.assertNotEqual(imm1, imm2)
        self.assertEqual(imm1, imm3)

    def test_var_equal(self):
        var1 = IntVar(values=[1, 256], name="var_1")
        var2 = IntVar(values=[1, 256], name="var_2")
        var3 = IntVar(values=[1, 256])

        var1_dup = IntVar(values=[1, 256], name="var_1")

        self.assertNotEqual(var1, var2)
        self.assertNotEqual(var1, var3)
        self.assertEqual(var1, var1_dup)

    def test_new_symbol(self):
        sym1 = symbolic.create_new_symbol(name="sym_1")  # noqa: F841
        # Create same symbol
        sym1_dup = symbolic.create_new_symbol(name="sym_1")  # noqa: F841
        # Capture error if 2 symbols share the same name but different value
        with self.assertRaises(ValueError):
            _ = symbolic.create_new_symbol(
                name="sym_1", values=[1, 256], check_duplicate=True
            )

        sym2 = symbolic.create_new_symbol(name="sym_2", values=[2, 32])  # noqa: F841
        sym2_dup = symbolic.create_new_symbol(  # noqa: F841
            name="sym_2", values=[2, 32]
        )
        with self.assertRaises(ValueError):
            _ = symbolic.create_new_symbol(
                name="sym_2", values=[1, 256], check_duplicate=True
            )

    def test_is_integer(self):
        self.assertTrue(symbolic.is_integer(3))
        self.assertFalse(symbolic.is_integer(3.5))
        self.assertFalse(symbolic.is_integer("string"))
        self.assertFalse(symbolic.is_integer([3, 4, 5]))

        sym1 = sympy.Symbol("sym_1")
        self.assertTrue(symbolic.is_integer(sym1 / sym1))
        sym2 = 2 * sym1
        self.assertTrue(symbolic.is_integer(sym2 / sym1))
        sym3 = 1.5 * sym1
        self.assertFalse(symbolic.is_integer(sym3 / sym1))
        self.assertTrue(symbolic.is_integer(sym3 / sym1 * 2))

    def test_elementwise_symbolic(self):
        var1 = IntVar(values=[1, 256], name="var_1")
        sym1 = var1.symbolic_value()
        var2 = IntVar(values=[1, 256], name="var_2")
        sym2 = var2.symbolic_value()

        tensor1 = IntVarTensor(int_var=var1)
        tensor2 = IntVarTensor(int_var=var2)

        add = ops.elementwise(FuncEnum.ADD)(tensor1, tensor2)
        self.assertEqual(add._attrs["symbolic_value"], sym1 + sym2)
        sub = ops.elementwise(FuncEnum.SUB)(tensor1, tensor2)
        self.assertEqual(sub._attrs["symbolic_value"], sym1 - sym2)
        mul = ops.elementwise(FuncEnum.MUL)(tensor1, tensor2)
        self.assertEqual(mul._attrs["symbolic_value"], sym1 * sym2)
        div = ops.elementwise(FuncEnum.DIV)(tensor1, tensor2)
        self.assertEqual(div._attrs["symbolic_value"], sym1 / sym2)


if __name__ == "__main__":
    unittest.main()
