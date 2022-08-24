# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Batched Gemm ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[b, m, n] = bmm(a[b, m, k], b[b, n, k])
This is used for `ops.bmm_rcr`.
"""

from ... import registry
from . import bmm_common, bmm_permute_common, bmm_rcr, common
from .layout import RCR


@registry.reg("rocm.bmm_rcr_permute.config")
def bmm_config(func_attrs, dtype="float16"):
    """Extract (operation name, operation instance) pair from
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

    op_kind = ck_lib.library.GemmKind.BatchGemmPermute
    extra_kind = ck_lib.library.TensorOperation.PassThrough
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.bmm_rcr_permute.gen_profiler")
def bmm_gen_profiler(func_attrs, workdir, dim_info_dict):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from bmm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    """
    extra_code = bmm_permute_common.EXTRA_PERM_ARGS_TEMPLATE.render(
        g1=func_attrs["shape"][0]
    )
    bmm_common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        bmm_rcr.ARGS_PARSER_TEMPLATE.render(),
        gemm_flag="",
        problem_args_template=bmm_permute_common.PROBLEM_ARGS_TEMPLATE,
        extra_header_template=bmm_permute_common.EXTRA_HEADER_TEMPLATE,
        extra_code=extra_code,
    )


@registry.reg("rocm.bmm_rcr_permute.gen_function")
def bmm_gen_function(func_attrs, exec_cond_template, dim_info_dict):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from bmm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    extra_code = bmm_permute_common.EXTRA_PERM_ARGS_TEMPLATE.render(
        g1=func_attrs["shape"][0]
    )
    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        gemm_flag="",
        problem_args_template=bmm_permute_common.PROBLEM_ARGS_TEMPLATE,
        extra_header_template=bmm_permute_common.EXTRA_HEADER_TEMPLATE,
        extra_code=extra_code,
    )


@registry.reg("rocm.bmm_rcr_permute.func_decl")
def bmm_gen_function_decl(func_attrs):
    """Generates function declarations.

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
    return bmm_common.gen_function_decl(func_name=func_name, gemm_flag="")


@registry.reg("rocm.bmm_rcr_permute.func_call")
def bmm_gen_function_call(func_attrs, indent="  "):
    """Generates function call.

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
    return bmm_common.gen_function_call(func_attrs, indent, gemm_flag="")


@registry.reg("rocm.bmm_rcr_permute.filter")
def gemm_function_filter(cfg, func_attrs, x_shape):
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
    return True
