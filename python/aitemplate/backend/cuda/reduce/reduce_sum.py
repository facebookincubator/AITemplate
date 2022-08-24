# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
reduce_sum kernel:
    (1) it invokes reduce_3d kernel for reduction_dim = -1 cases; and
    (2) invokes reduce_common for all other cases

We do this because there is huge perf difference between reduce_3d and
reduce_common for different reduction dims. We should consider to unify
our implementation later. Ideally, we should fix the perf issue in
reduce_3d for non-neg-dim cases, because reduce_3d can take prologue and
epilogue so it is more general than reduce_common.
"""

from ... import registry
from . import reduce_3d, reduce_common


def _is_last_reduction_dim(func_attrs):
    """return true if the reduction dim is the last dim (i.e. inner most dim)"""
    axes = func_attrs["reduction_axes"]
    if not len(axes) == 1:
        raise NotImplementedError("Multiple reduction axes are not supported yet")
    reduction_dim = axes[0]
    # make sure our frontend handle negative dims
    assert reduction_dim >= 0, "cannot have negative dim here: {}".format(reduction_dim)
    x = func_attrs["inputs"][0]
    rank = x._rank()
    assert rank >= 1, "rank must >= 1, got: {}".format(rank)
    return reduction_dim == rank - 1


@registry.reg("cuda.reduce_sum.func_decl")
def reduce_sum_gen_function_decl(func_attrs):
    """the registered function for generating reduce_sum function declaration

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_sum op

    Returns
    -------
    [type] : str
        returns the rendered function declaration with appropriate replacements
    """
    if _is_last_reduction_dim(func_attrs):
        return reduce_3d.gen_function_decl(func_attrs)
    else:
        return reduce_common.gen_function_decl(func_attrs)


@registry.reg("cuda.reduce_sum.gen_function")
def reduce_sum_gen_function(func_attrs):
    """the registered function for generating reduce_sum kernel and all of
    its auxiliary functions

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_sum op

    Returns
    -------
    str
        returns the rendered code for the complete implementation of this reduce sum op
    """
    if _is_last_reduction_dim(func_attrs):
        return reduce_3d.gen_function(func_attrs, "cutlass::plus")
    else:
        return reduce_common.gen_function(func_attrs, "cutlass::plus")


@registry.reg("cuda.reduce_sum.func_call")
def reduce_sum_gen_function_call(func_attrs, indent="  "):
    """the registered function for generating a function call to reduce_mean

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_mean op
    indent : str, optional
        indentation for each line of the rendered code (default "  ")

    Returns
    -------
    str
        returns rendered code for invoking the reduce op
    """
    if _is_last_reduction_dim(func_attrs):
        return reduce_3d.gen_function_call(func_attrs, indent)
    else:
        return reduce_common.gen_function_call(func_attrs, indent)
