# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
conv2d bias relu codegen
"""
from ... import registry
from . import common, common_conv2d_bias_activation as cba

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("cuda.conv2d_bias_relu.config")
def conv2d_config(func_attrs, dtype="float16"):
    """[summary]

    Parameters
    ----------
    cutlass_path : [type]
        [description]
    dtype : str, optional
        [description], by default "float16"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    func_attrs["op_instance"] = common.extract_config(func_attrs)


@registry.reg("cuda.conv2d_bias_relu.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    """[summary]

    Parameters
    ----------
    template_path : [type]
        [description]
    workdir : [type]
        [description]
    shape_template : [type]
        [description]
    """
    cba.gen_profiler(func_attrs, workdir, shape_template)


@registry.reg("cuda.conv2d_bias_relu.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    """[summary]

    Parameters
    ----------
    template_path : [type]
        [description]
    func_name : [type]
        [description]
    exec_path : [type]
        [description]
    exec_cond_remplate : [type]
        [description]
    shape_eval_template : [type]
        [description]
    shape_save_template : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return common.gen_function(
        func_attrs,
        cba.INSTANCE_TEMPLATE,
        cba.EXEC_TEMPLATE,
        cba.SRC_TEMPLATE,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
    )


@registry.reg("cuda.conv2d_bias_relu.func_decl")
def conv2d_gen_function_decl(func_attrs):
    """[summary]

    Parameters
    ----------
    func_name : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    func_name = func_attrs["name"]
    return cba.FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.conv2d_bias_relu.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    indent : str, optional
        [description], by default "  "

    Returns
    -------
    [type]
        [description]
    """
    return cba.gen_function_call(func_attrs, indent)


@registry.reg("cuda.conv2d_bias_relu.filter")
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
    return common.function_filter(cfg, func_attrs, x_shape)
