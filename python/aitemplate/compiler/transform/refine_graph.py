# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from typing import List

from ..base import IntVar, Operator, Tensor

# pylint: disable=C0103

SPECIAL_CHECK_FUNC_KEYS = {"inputs", "name", "depth", "outputs", "gemm_operand_groups"}


def same_int_var(v1: IntVar, v2: IntVar):
    """[summary]

    Parameters
    ----------
    v1 : IntVar
        [description]
    v2 : IntVar
        [description]

    Returns
    -------
    [type]
        [description]
    """
    v1v = v1._attrs["values"]
    v2v = v2._attrs["values"]
    if len(v1v) != len(v2v):
        return False
    for s1, s2 in zip(v1v, v2v):
        if s1 != s2:
            return False
    return True


def same_tensor_type(t1: Tensor, t2: Tensor):
    """[summary]

    Parameters
    ----------
    t1 : Tensor
        [description]
    t2 : Tensor
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if t1.dtype() != t2.dtype():
        return False
    t1s = t1.shape()
    t2s = t2.shape()
    if len(t1s) != len(t2s):
        return False
    for d1, d2 in zip(t1s, t2s):
        if not same_int_var(d1, d2):
            return False
    return True


def same_function_type(o1: Operator, o2: Operator):
    """[summary]

    Parameters
    ----------
    o1 : Operator
        [description]
    o2 : Operator
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if len(o1._attrs) != len(o2._attrs):
        return False
    keys = o1._attrs.keys()
    # check general attrs
    for key in keys:
        if key not in o2._attrs:
            return False
        if key not in SPECIAL_CHECK_FUNC_KEYS:
            if o1._attrs[key] != o2._attrs[key]:
                return False
    # check inputs
    o1_args = o1._attrs["inputs"]
    o2_args = o2._attrs["inputs"]
    if len(o1_args) != len(o2_args):
        return False
    for t1, t2 in zip(o1_args, o2_args):
        if not same_tensor_type(t1, t2):
            return False
    return True


def refine_graph(sorted_graph: List[Tensor]):
    """[summary]

    Parameters
    ----------
    sorted_graph : List[Tensor]
        [description]
    """
    exist_func = []
    for node in sorted_graph:
        for func in node.src_ops():
            found = False
            for f in reversed(exist_func):
                if same_function_type(f, func):
                    func._attrs["name"] = f._attrs["name"]
                    found = True
                    break
            if not found:
                exist_func.append(func)
