# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from ... import registry
from ..gemm_universal import common
from . import common_softmax, gemm_rcr_softmax


# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    /*
        A: M*K (RowMajor)
        B: N*K (ColumnMajor)
        C/D/sofmax: M*N (RowMajor)
        N: M*1 (RowMajor)
    */

        {M, N, K},
        1,
        {a_ptr, LayoutA(K)},
        {b_ptr, LayoutB(K)},
        {c_ptr, 0},
        {d_ptr, LayoutC(N)},
        {
            float(1.0),
            float(1.0)
        },
        {n_ptr, LayoutC(1)},
        {soft_ptr, LayoutC(N)}

"""
)


@registry.reg("cuda.gemm_rcr_bias_softmax.config")
def gemm_rcr_bias_softmax_config(func_attrs, dtype="float16"):
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
    return gemm_rcr_softmax.gemm_rcr_softmax_config(func_attrs, dtype)


@registry.reg("cuda.gemm_rcr_bias_softmax.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    return gemm_rcr_softmax.common_gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common_softmax.SRC_TEMPLATE,
        PROBLEM_ARGS_TEMPLATE,
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    return gemm_rcr_softmax.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        PROBLEM_ARGS_TEMPLATE,
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.func_decl")
def gen_function_decl(func_attrs):
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return gemm_rcr_softmax.gen_function_decl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_softmax.func_call")
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
    return gemm_rcr_softmax.gen_function_call(
        func_attrs,
        indent,
    )


@registry.reg("cuda.gemm_rcr_bias_softmax.filter")
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
