# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import jinja2

from .. import registry

# pylint: disable=C0301

VAR_TEMPLATE = jinja2.Template("""{{indent}} int64_t {{name}} { {{value}} };""")

PTR_TEMPLATE = jinja2.Template("""{{indent}} {{dtype}} {{name}} {nullptr};""")


@registry.reg("cuda.lib.var_decl")
def var_decl(name, value=0, indent="  "):
    """[summary]

    Parameters
    ----------
    name : [type]
        [description]
    value : int, optional
        [description], by default 0
    indent : str, optional
        [description], by default "  "

    Returns
    -------
    [type]
        [description]
    """
    return VAR_TEMPLATE.render(name=name, value=value, indent=indent)


@registry.reg("cuda.lib.ptr_decl")
def ptr_decl(name, dtype="float16", indent="  "):
    """[summary]

    Parameters
    ----------
    name : [type]
        [description]
    dtype : str, optional
        [description], by default "float16"
    indent : str, optional
        [description], by default "  "

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    if dtype == "float16":
        type_string = "cutlass::half_t*"
    elif dtype in ["float", "float32"]:
        type_string = "float*"
    elif dtype == "int64":
        type_string = "int64_t*"
    elif dtype in ["int", "int32"]:
        type_string = "int32_t*"
    else:
        raise NotImplementedError
    return PTR_TEMPLATE.render(name=name, dtype=type_string, indent=indent)
