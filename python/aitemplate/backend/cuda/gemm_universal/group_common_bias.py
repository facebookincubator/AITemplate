# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from . import group_common

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
        problem_sizes_device,
        problem_count,
        threadblock_count,
        {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
        ptr_A,
        ptr_B,
        ptr_bias,
        ptr_C,
        lda,
        ldb,
        ldc,
        ldd
"""
)


def gen_profiler(
    func_attrs,
    workdir,
    shape_template,
):
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
    group_common.gen_profiler(
        func_attrs, workdir, shape_template, PROBLEM_ARGS_TEMPLATE, has_bias=True
    )


def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    exec_cond_template : [type]
        [description]
    shape_eval_template : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return group_common.gen_function(
        func_attrs,
        exec_cond_template,
        shape_eval_template,
        PROBLEM_ARGS_TEMPLATE,
        has_bias=True,
    )


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
    func_name = func_attrs["name"]
    return group_common.FUNC_DECL_TEMPLATE.render(
        func_name=func_name, groups=func_attrs["groups"], has_bias=True
    )


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
    ndims = 3
    return group_common.gen_function_call(func_attrs, ndims, has_bias=True)
