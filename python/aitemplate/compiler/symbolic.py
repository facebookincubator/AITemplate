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
Symbolic helpers for AITemplate.
AITemplate leverages Sympy to do symbolic computations for shapes.
The core of Sympy is surrounded around the class "Symbol". We could apply operations
on Symbols (i.e. add/mul/power/etc.) Which could help us do basic arithmetic with
unknown values.
The symbolic-ness comes from representation that includes Symbol (i.e. sym_1 + 100.)

Example Usage:
A = IntVar(...)
sym_A = A.symbolic_value() # equivalent of A._attrs["symbolic_value"]

# do something about sym_A, some common usage include:
new_sym = sym_A + 100
new_sym = sym_A - 100
new_sym = sym_A * 2
new_sym = sym_A * sym_B

# We could then assign the symbolic value to a new IntVar.
new_var = IntVar(..., symbolic_value=new_sym)

For more advanced usage on Sympy, check: https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
"""
from __future__ import annotations

import itertools

from numbers import Number
from typing import Any, List, Optional, Set

import sympy


_k_symbolic_to_intvar = {}
_k_symbolic_index = 0
_k_symbolic_value = {}


def create_new_symbol(
    name: Optional[str] = None,
    values: Optional[List[int]] = None,
    check_duplicate: bool = False,
) -> sympy.Symbol:
    """
    Creates and memoizing symbols.

    Parameters
    ----------
    name : Optional[str]
        The symbol name that is going to be used. If None is provided, an unused
        name would be created.
    values : Optional[List[int]]
        The values for IntVar, which indicates the range of which the symbol could
        represent.
    check_duplicate : bool
        If set as True and name is provided, we check whether the name and values
        provided matches the corresponding symbol recorded.
    """
    global _k_symbolic_index
    global _k_symbolic_value

    if name is None:
        while True:
            name = f"_sym_{_k_symbolic_index}"
            _k_symbolic_index += 1

            if name not in _k_symbolic_value:
                break

    values = sorted(set(values)) if values is not None else values
    if (
        check_duplicate
        and name in _k_symbolic_value
        and _k_symbolic_value[name] != values
    ):
        raise ValueError(
            f"Symbol ({name}) has different values! New value is {values}, stored value is {_k_symbolic_value[name]}"
        )

    _k_symbolic_value[name] = values
    return sympy.Symbol(name)


def is_symbol(sym_val: Any) -> bool:
    return isinstance(sym_val, sympy.Symbol)


def is_symbolic(sym_val: Any) -> bool:
    """
    Check whether sym_val is a sympy class.
    """
    return isinstance(sym_val, sympy.Basic)


def is_integer(sym_val: Any) -> bool:
    # We wrap this since None is returned if sympy can't determine the property.
    if is_symbolic(sym_val):
        return sym_val.is_number and int(sym_val) - sym_val == 0
    elif isinstance(sym_val, Number):
        return int(sym_val) - sym_val == 0

    return False


def get_global_symbol_set() -> Set:
    global _k_symbolic_value
    return set(_k_symbolic_value.keys())


def get_intvar(sym_name: str):
    global _k_symbolic_to_intvar

    return _k_symbolic_to_intvar.get(sym_name, None)


def store_intvar(sym_name: str, int_var) -> None:
    global _k_symbolic_to_intvar

    _k_symbolic_to_intvar[sym_name] = int_var


def simplify_intvar_values(sym_val: sympy.Basic):
    """
    Given a symbolic value, resolve the symbol's value range.

    Example:
    'symbol_A' has value range of [10, 20]
    simplify_intvar_values(symbol_A * 3 + 4) returns [34, 64]
    """
    global _k_symbolic_value

    symbols = list(sym_val.free_symbols)
    symbol_shapes = [_k_symbolic_value[s.name] for s in symbols]
    symbol_shapes = [s for s in symbol_shapes if s is not None]
    shape_perms = list(itertools.product(*symbol_shapes))

    new_shape = [int(sym_val.subs(zip(symbols, s))) for s in shape_perms]
    new_shape = sorted(set(new_shape))

    return new_shape
