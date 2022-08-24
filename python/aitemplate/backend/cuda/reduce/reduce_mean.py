# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
A reduce_mean kernel implementation
"""

import jinja2

from ... import registry
from . import reduce_3d


EPILOGUE_SCALAR_TEMPLATE = jinja2.Template(
    """
{{indent}}return (reduced_result / ElementCompute(num_reduced_elems));
"""
)


@registry.reg("cuda.reduce_mean.func_decl")
def reduce_mean_gen_function_decl(func_attrs) -> str:
    """the registered function for generating reduce_mean function declaration

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_mean op

    Returns
    -------
    str
        returns the rendered function declaration with appropriate replacements
    """
    return reduce_3d.gen_function_decl(func_attrs)


@registry.reg("cuda.reduce_mean.gen_function")
def reduce_mean_gen_function(func_attrs) -> str:
    """the registered function for generating reduce_mean kernel and all of
    its auxiliary functions

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds attributes of this reduce_mean op

    Returns
    -------
    str
        returns the rendered code for the complete implementation of this reduce mean op
    """
    return reduce_3d.gen_function(
        func_attrs,
        "cutlass::plus",
        reduce_3d.DEFAULT_PROLOGUE_TEMPLATE,
        EPILOGUE_SCALAR_TEMPLATE,
    )


@registry.reg("cuda.reduce_mean.func_call")
def reduce_mean_gen_function_call(func_attrs, indent="  ") -> str:
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
    return reduce_3d.gen_function_call(func_attrs, indent)
