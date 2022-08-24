# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Util functions to handle shapes.
"""

from typing import List, Tuple

from aitemplate.compiler.base import IntImm, IntVar, IntVarTensor


def gen_int_var(values: List[int], name: str = None) -> IntVar:
    """
    A helper function to generate IntImm or IntVar depending on the length of values.
    """

    values = list(set(values))
    if len(values) == 1:
        return IntImm(values[0], name=name)
    elif len(values) > 1:
        return IntVar(values, name=name)
    else:
        raise RuntimeError("Unsupported dim definition: {}".format(values))


def gen_int_var_min_max(values: List[int], name: str = None) -> IntVar:
    """
    A helper function to generate IntImm or IntVar depending on the length of values.
    Only keeps [min, max] pairs if there are more than 2 values.
    """
    return gen_int_var([min(values), max(values)], name=name)


def get_broadcast_max_shape(
    shape1: List[IntVar], shape2: List[IntVar]
) -> Tuple[bool, List[IntVar]]:
    """
    Checks whether two inputs shapes are broadcastable, and if yes, also returns the result broadcast shape.
    Two shapes are broadcastable if starting from trailing (rightmost) dimensions both dims are:
        1. equal, or
        2. one of them is 1
    Note that two shapes are not required to have the same number of dimensions.
    For example, shape [5, 2, 3] and shape [3] are also broadcastable.
    """

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


def get_num_rightmost_static_elements(
    shape: List[IntVar], num_rightmost_dims: int = None
) -> int:
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

    res = 1

    for idx, dim in enumerate(reversed(shape)):
        if idx >= num_rightmost_dims:
            break
        if not isinstance(dim, IntImm):
            break
        res *= dim._attrs["values"][0]
    return res


def all_static_dimensions(shape: List[IntVar], from_dim: int = 0):
    """
    Return true if all dimensions starting from from_dim (inclusive)
    are static
    """
    for dim in shape[from_dim:]:
        if not isinstance(dim, IntImm):
            return False
    return True


def is_static_dimension(shape: List[IntVar], dim: int) -> bool:
    """
    Return true if shape[dim] is static
    """
    return dim <= len(shape) and isinstance(shape[dim], IntImm)


def convert_shape_to_IntVar(shape):
    """
    Helper function to convert a list of mixed int/IntVar/IntImm
    into a list with only IntVar/IntImm.
    """
    ret = []
    for v in shape:
        if isinstance(v, int):
            ret.append(IntImm(v))
        elif isinstance(v, IntVar):
            ret.append(v)
        elif isinstance(v, IntVarTensor):
            ret.append(v._attrs["int_var"])
    return ret
