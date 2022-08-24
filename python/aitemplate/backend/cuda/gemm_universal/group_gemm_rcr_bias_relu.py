# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
from ... import registry
from . import common, group_common_bias, group_gemm_rcr

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


@registry.reg("cuda.group_gemm_rcr_bias_relu.config")
def group_rcr_config(func_attrs, dtype="float16"):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
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
    group_gemm_rcr.group_rcr_config(func_attrs, dtype)


@registry.reg("cuda.group_gemm_rcr_bias_relu.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    """_summary_

    Parameters
    ----------
    func_attrs: _type_
        _description_
    workdir : _type_
        _description_
    shape_template : _type_
        _description_
    """
    group_common_bias.gen_profiler(func_attrs, workdir, shape_template)


@registry.reg("cuda.group_gemm_rcr_bias_relu.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    exec_cond_remplate : [type]
        [description]
    shape_eval_template : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return group_common_bias.gen_function(
        func_attrs,
        exec_cond_remplate,
        shape_eval_template,
    )


@registry.reg("cuda.group_gemm_rcr_bias_relu.func_decl")
def gen_function_decl(func_attrs):
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
    return group_common_bias.gen_function_decl(func_attrs)


@registry.reg("cuda.group_gemm_rcr_bias_relu.func_call")
def gen_function_call(func_attrs, indent="  "):
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
    return group_common_bias.gen_function_call(func_attrs, indent)


@registry.reg("cuda.group_gemm_rcr_bias_relu.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
