# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
C[m, b, n](row) = bmm(A[m, b, k](row), B[b, n, k](col))
in torch it is
# _2905_2929 = _2904.view(B, 25, -1).permute(1, 0, 2)
# _2930_2954 = torch.baddbmm(
#      self._1085_1133, _2905_2929, self._1084_1132) # baddbmm(bias, X, W)
"""
from ... import registry
from . import bmm_common, common

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_problem_info(**kwargs):
    problem_args = {
        "bias_ptr": "c_ptr",
        "a_batch_stride": "K",
        "b_batch_stride": "N * K",
        "bias_batch_stride": "N",
        "c_batch_stride": "N",
        "lda": "K * B",
        "ldb": "K",
        "ldbias": "N * B",
        "ldc": "N * B",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


@registry.reg("cuda.perm102_bmm_rcr.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    def fproc_f16(op):
        import cutlass_lib

        return common.default_fproc_f16(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.RowMajor,
            b_layout=cutlass_lib.library.LayoutType.ColumnMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            epiligue_name=func_attrs["epilogue"],
        )

    func_attrs["op_instance"] = common.extract_config(fproc_f16)


@registry.reg("cuda.perm102_bmm_rcr.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=["M", "B", "K"], b_dims=["B", "N", "K"], c_dims=["M", "B", "N"]
    )

    mm_info = _get_problem_info(alpha_value=func_attrs.get("alpha", 1))
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(mm_info=mm_info)

    bmm_common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common.SRC_TEMPLATE,
        problem_args,
        args_parser,
    )


@registry.reg("cuda.perm102_bmm_rcr.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    mm_info = _get_problem_info(alpha_value=func_attrs.get("alpha", 1))
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(mm_info=mm_info)

    return bmm_common.gen_function(
        func_attrs,
        exec_cond_template,
        problem_args,
        dim_info_dict,
    )


@registry.reg("cuda.perm102_bmm_rcr.func_decl")
def gen_function_decl(func_attrs):
    return bmm_common.gen_function_decl(func_attrs)


@registry.reg("cuda.perm102_bmm_rcr.func_call")
def gen_function_call(func_attrs, indent="  "):
    return bmm_common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.perm102_bmm_rcr.filter")
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
