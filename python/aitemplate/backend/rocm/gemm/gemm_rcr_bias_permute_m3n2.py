# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM ROCM backend for A[RowMajor], B[ColumnMajor], C[RowMajor], i.e.
c[m, n] = a[m, k] * b[n, k] + bias[n]
where m = M0 * M1 * M2, n = N0 * N1
c = c.reshape(M0, M1, M2, N0, N1)
output = torch.permute(c, [2, 0, 3, 1, 4])
"""
from ... import registry
from . import common, permute_common
from .layout import RCR


@registry.reg("rocm.gemm_rcr_bias_permute_m3n2.config")
def gemm_config(func_attrs, dtype="float16"):
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
    import ck_lib  # noqa: F401

    op_kind = ck_lib.library.GemmKind.GemmPermuteM3N2
    extra_kind = ck_lib.library.TensorOperation.Add
    common.make_fproc_f16(func_attrs, RCR, op_kind, extra_kind)


@registry.reg("rocm.gemm_rcr_bias_permute_m3n2.gen_profiler")
def gemm_gen_profiler(func_attrs, workdir, dim_info_dict):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    """
    common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        dim_info_dict=dim_info_dict,
        args_parse=RCR.args_parse,
        gemm_flag="bias_permute_m3n2",
        extra_code="const int G1={}, G2={}, G3={};".format(
            func_attrs["shape"][0],
            func_attrs["shape"][1],
            func_attrs["shape"][2],
        ),
        extra_shape_template=permute_common.EXTRA_SHAPE_TEMPLATE_M3N2,
    )


@registry.reg("rocm.gemm_rcr_bias_permute_m3n2.gen_function")
def gemm_gen_function(func_attrs, exec_cond_template, dim_info_dict):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        "bias_permute_m3n2",
        extra_code="const int G1={}, G2={}, G3={};".format(
            func_attrs["shape"][0],
            func_attrs["shape"][1],
            func_attrs["shape"][2],
        ),
        extra_shape_template=permute_common.EXTRA_SHAPE_TEMPLATE_M3N2,
    )


@registry.reg("rocm.gemm_rcr_bias_permute_m3n2.func_decl")
def gemm_gen_function_decl(func_attrs):
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
    return common.gen_function_decl(func_name=func_name, gemm_flag="bias")


@registry.reg("rocm.gemm_rcr_bias_permute_m3n2.func_call")
def gemm_gen_function_call(func_attrs, indent="  "):
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
    return common.gen_function_call(func_attrs, indent, gemm_flag="bias")


@registry.reg("rocm.gemm_rcr_bias_permute_m3n2.filter")
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
