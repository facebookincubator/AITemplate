# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[RowMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When used for `linear`, need to set A->Data, B->Weight
"""
from ... import registry
from ...common import gemm_common
from . import bmm_common, common

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_problem_info(**kwargs):
    problem_args = {
        "bias_ptr": "c_ptr",
        "a_batch_stride": "M * K",
        "b_batch_stride": "N * K",
        "bias_batch_stride": "M * N",
        "c_batch_stride": "M * N",
        "lda": "K",
        "ldb": "N",
        "ldbias": "N",
        "ldc": "N",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


@registry.reg("cuda.bmm_rrr.config")
def bmm_rrr_config(func_attrs, dtype="float16"):
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
            a_layout=cutlass_lib.library.LayoutType.RowMajor,
            b_layout=cutlass_lib.library.LayoutType.RowMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            epiligue_name=func_attrs["epilogue"],
        )

    func_attrs["op_instance"] = common.extract_config(fproc_f16)


@registry.reg("cuda.bmm_rrr.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    a_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.INPUT, 0
    )
    b_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.INPUT, 1
    )
    c_dims = bmm_common.reverse_dim_info_mapping(
        dim_info_dict, gemm_common.Source.OUTPUT, 0
    )

    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=a_dims, b_dims=b_dims, c_dims=c_dims
    )

    mm_info = _get_problem_info(alpha_value=func_attrs.get("alpha", 1))
    a_shapes = func_attrs["input_accessors"][0].original_shapes
    b_shapes = func_attrs["input_accessors"][1].original_shapes
    bmm_common._update_stride_info(mm_info, a_shapes, b_shapes)

    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(mm_info=mm_info)

    bmm_common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common.SRC_TEMPLATE,
        problem_args,
        args_parser,
    )


@registry.reg("cuda.bmm_rrr.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    mm_info = _get_problem_info(alpha_value=func_attrs.get("alpha", 1))
    a_shapes = func_attrs["input_accessors"][0].original_shapes
    b_shapes = func_attrs["input_accessors"][1].original_shapes
    bmm_common._update_stride_info(mm_info, a_shapes, b_shapes)

    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(mm_info=mm_info)

    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
    )


@registry.reg("cuda.bmm_rrr.func_decl")
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
    return bmm_common.gen_function_decl(func_attrs)


@registry.reg("cuda.bmm_rrr.func_call")
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


@registry.reg("cuda.bmm_rrr.filter")
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
