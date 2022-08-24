# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] rocm concatenate function
"""

from ... import registry
from ...backend_spec import ROCMSpec
from ...common import concatenate_common


@registry.reg("rocm.concatenate.func_decl")
def gen_function_decl(func_attrs):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    Returns
    -------
    str
        Rendered function declaration.
    """
    return concatenate_common.gen_function_decl(
        func_attrs=func_attrs,
        backend_spec=ROCMSpec(),
    )


@registry.reg("rocm.concatenate.gen_function")
def gen_function(func_attrs, element_func=None, element_func_def=None):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.

    Returns
    -------
    str
        Rendered function body.
    """
    return concatenate_common.gen_function(
        func_attrs=func_attrs,
        backend_spec=ROCMSpec(),
        element_func=element_func,
        element_func_def=element_func_def,
    )


@registry.reg("rocm.concatenate.func_call")
def gen_function_call(func_attrs, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    return concatenate_common.gen_function_call(
        func_attrs=func_attrs,
        backend_spec=ROCMSpec(),
        indent="  ",
    )
