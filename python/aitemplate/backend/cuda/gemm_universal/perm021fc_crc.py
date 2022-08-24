# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
from ... import registry
from . import bmm_common, common

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_problem_info(**kwargs):
    problem_args = {
        "problem_size": "{N, M, K}",
        "bias_ptr": "c_ptr",
        "a_batch_stride": "0",
        "b_batch_stride": "K * M",
        "bias_batch_stride": "M * N",
        "c_batch_stride": "M * N",
        "lda": "N",
        "ldb": "M",
        "ldbias": "N",
        "ldc": "N",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


@registry.reg("cuda.perm021fc_crc.config")
def gemm_crc_config(func_attrs, dtype="float16"):
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

    def fproc_f16(op):
        import cutlass_lib

        return common.default_fproc_f16(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            b_layout=cutlass_lib.library.LayoutType.RowMajor,
            c_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            epiligue_name=func_attrs["epilogue"],
        )

    func_attrs["op_instance"] = common.extract_config(fproc_f16)


@registry.reg("cuda.perm021fc_crc.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=["1", "K", "N"], b_dims=["B", "K", "M"], c_dims=["B", "M", "N"]
    )

    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=_get_problem_info(alpha_value=func_attrs.get("alpha", 1), beta_value=0)
    )

    bmm_common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common.SRC_TEMPLATE,
        problem_args,
        args_parser,
    )


@registry.reg("cuda.perm021fc_crc.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=_get_problem_info(alpha_value=func_attrs.get("alpha", 1), beta_value=0)
    )

    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
    )


@registry.reg("cuda.perm021fc_crc.func_decl")
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
    return bmm_common.gen_function_decl(func_attrs)


@registry.reg("cuda.perm021fc_crc.func_call")
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
    return bmm_common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.perm021fc_crc.filter")
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
