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
Util functions to handle shapes.
"""

from typing import List, Optional

import sympy


def gen_int_var(
    values: List[int], name: str = None, symbolic_value: Optional[sympy.Basic] = None
):
    """
    A helper function to generate IntImm or IntVar depending on the length of values.
    """
    from aitemplate.compiler.base import IntImm, IntVar

    values = list(set(values))
    if len(values) == 1:
        return IntImm(values[0], name=name)
    elif len(values) > 1:
        return IntVar(values, name=name, symbolic_value=symbolic_value)
    else:
        raise RuntimeError("Unsupported dim definition: {}".format(values))


def gen_int_var_min_max(
    values: List[int], name: str = None, symbolic_value: Optional[sympy.Basic] = None
):
    """
    A helper function to generate IntImm or IntVar depending on the length of values.
    Only keeps [min, max] pairs if there are more than 2 values.
    """
    return gen_int_var(
        [min(values), max(values)], name=name, symbolic_value=symbolic_value
    )


def get_broadcast_max_shape(shape1, shape2):
    """
    Checks whether two inputs shapes are broadcastable, and if yes, also returns the result broadcast shape.
    Two shapes are broadcastable if starting from trailing (rightmost) dimensions both dims are:
        1. equal, or
        2. one of them is 1
    Note that two shapes are not required to have the same number of dimensions.
    For example, shape [5, 2, 3] and shape [3] are also broadcastable.
    """
    from aitemplate.compiler.base import IntImm

    min_len = min(len(shape1), len(shape2))
    if len(shape1) > len(shape2):
        res_shape = list(shape1)
    else:
        res_shape = list(shape2)
    for i in range(min_len):
        idx = -i - 1
        dim1 = shape1[idx]
        dim2 = shape2[idx]
        if dim1 == dim2:
            res_shape[idx] = dim1
            continue
        if dim1 == IntImm(1):
            res_shape[idx] = dim2
        elif dim2 == IntImm(1):
            res_shape[idx] = dim1
        else:
            return (False, None)
    return (True, res_shape)


def get_num_rightmost_static_elements(shape, num_rightmost_dims: int = None) -> int:
    """
    Returns number of elements in rightmost max contiguous static dimensions.
    If the rightmost dim is dynamic, returns 1.

    If num_rightmost_dims is specified, only look into num_rightmost_dims.
    Otherwise, look into all dims.
    If num_rightmost_dims == 0, returns 1.

    This is useful when calculating alignment.

    e.g.
    shape = [IntImm(2), IntImm(4)] returns 2 * 4 = 8.
    shape = [IntImm(2), IntVar(4, 8)] returns 1.
    shape = [IntImm(9), IntVar(4, 8), IntImm(3)] returns 3.

    shape = [IntImm(2), IntImm(4), IntImm(3)], num_rightmost_dims = None returns 24.
    shape = [IntImm(2), IntImm(4), IntImm(3)], num_rightmost_dims = 1 returns 3.
    shape = [IntImm(2), IntImm(4), IntImm(3)], num_rightmost_dims = 0 returns 1.
    """
    from aitemplate.compiler.base import IntImm

    res = 1

    for idx, dim in enumerate(reversed(shape)):
        if num_rightmost_dims is not None and idx >= num_rightmost_dims:
            break
        if not isinstance(dim, IntImm):
            break
        res *= dim._attrs["values"][0]
    return res


def all_static_dimensions(shape, from_dim: int = 0):
    """
    Return true if all dimensions starting from from_dim (inclusive)
    are static
    """
    from aitemplate.compiler.base import IntImm

    for dim in shape[from_dim:]:
        if not isinstance(dim, IntImm):
            return False
    return True


def is_static_dimension(shape, dim: int) -> bool:
    """
    Return true if shape[dim] is static
    """
    from aitemplate.compiler.base import IntImm

    return dim <= len(shape) and isinstance(shape[dim], IntImm)


def convert_shape_to_IntVar(shape):
    """
    Helper function to convert a list of mixed int/IntVar/IntImm
    into a list with only IntVar/IntImm.
    """
    from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor

    ret = []
    for v in shape:
        if isinstance(v, int):
            ret.append(IntImm(v))
        elif isinstance(v, IntVar):
            ret.append(v)
        elif isinstance(v, IntVarTensor):
            ret.append(v._attrs["int_var"])
    return ret


def convert_IntVar_to_int(var) -> int:
    """
    Try to convert an IntVar (or an IntVar wrapped in a IntVarTensor) to
    an int. Raises a value error if var is dynamic.
    """
    from aitemplate.compiler.base import IntVarTensor

    if isinstance(var, int):
        return var

    var = var._attrs["int_var"] if isinstance(var, IntVarTensor) else var
    if var.upper_bound() == var.lower_bound():
        return var.upper_bound()

    raise ValueError(f"Cannot convert IntVar to int: {var}")


def is_singleton_dimension(dim) -> bool:
    """
    True if this dimension is 1. IntVars will return True if their
    upper and lower bounds are both 1.
    """
    from aitemplate.compiler.base import IntVarTensor

    if isinstance(dim, int):
        return dim == 1

    dim = dim._attrs["int_var"] if isinstance(dim, IntVarTensor) else dim
    return dim.upper_bound() == dim.lower_bound() and dim.upper_bound() == 1


def is_same_shape(shapes1, shapes2) -> bool:
    if len(shapes1) != len(shapes2):
        return False
    for dim1, dim2 in zip(shapes1, shapes2):
        if dim1 != dim2:
            return False
    return True


def get_static_stride(shape, dim) -> Optional[int]:
    """
    This is a helper function that returns the static stride for dim.
    It returns None if it cannot generate a static stride.
    """
    from aitemplate.compiler.base import IntImm

    stride = 1
    for d in shape[dim + 1 :]:
        if not isinstance(d, IntImm):
            return None
        stride *= d.value()
    return stride
