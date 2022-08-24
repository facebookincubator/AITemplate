# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ... import registry
from . import common

# pylint: disable=C0103,C0415,W0613


@registry.reg("rocm.conv2d.config")
def conv2d_config(func_attrs):
    """[summary]
    Extract (operation name, operation instance) pair from
    all operation candidates.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    Dict
        Extracted (operation name, operation instance) pair
        from all operation candidates.
    """
    import ck_lib

    op_kind = ck_lib.library.Conv2dKind.GroupConv2dBiasRelu
    extra_kind = ck_lib.library.TensorOperation.PassThrough
    func_attrs["op_instance"] = common.extract_config(op_kind, extra_kind)


@registry.reg("rocm.conv2d.gen_profiler")
def conv2d_gen_profiler(func_attrs, workdir, shape_template):
    """[summary]
    Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    shape_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    """
    common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        shape_template=shape_template,
        conv2d_flag="",
    )


@registry.reg("rocm.conv2d.gen_function")
def conv2d_gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    """[summary]
    Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    instance_template : jinja2.Template
        Template that defines the model. e.g. 'using model=xxx'.
    exec_template : jinja2.Template
        Execution statements in main function.
    src_template : jinja2.Template
        Full main.cpp with headers, embedding all templates.
    exec_cond_remplate : jinja2.Template
        Generates if statement to execute kernel.
    shape_eval_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    shape_save_template : jinja2.Template
        Generates output dimensions.
        The template is passed from compiler/ops/pool.


    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return common.gen_function(
        func_attrs, exec_cond_remplate, shape_eval_template, shape_save_template, ""
    )


@registry.reg("rocm.conv2d.func_decl")
def conv2d_gen_function_decl(func_attrs):
    """[summary]
    Generates function declarations.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    str
        The rentered template of function declaration.
    """
    func_name = func_attrs["name"]
    return common.gen_function_decl(func_name=func_name, conv2d_flag="")


@registry.reg("rocm.conv2d.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    """[summary]
    Generates function call.

    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    return common.gen_function_call(func_attrs, indent, conv2d_flag="")


@registry.reg("rocm.conv2d.filter")
def conv2d_function_filter(cfg, func_attrs, x_shape):
    """[summary]
    Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    # return common.function_filter(cfg, func_attrs, 1)
    return True
