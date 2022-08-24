# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
C[m, b, n](row) = bmm(A[m, b, k](row), B[b, k, n](row))
in torch it is
# _2905_2929 = _2904.view(B, 25, -1).permute(1, 0, 2)
# _2930_2954 = torch.baddbmm(
#      self._1085_1133, _2905_2929, self._1084_1132) # baddbmm(bias, X, W)
"""
from ... import registry
from . import bmm_common, common, common_bias, perm102_bmm_rrr

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_problem_info(**kwargs):
    problem_args = {
        "beta_value": 1,
        "bias_ptr": "bias_ptr",
        "a_batch_stride": "K",
        "b_batch_stride": "N * K",
        "bias_batch_stride": "N",
        "c_batch_stride": "N",
        "lda": "K * B",
        "ldb": "N",
        "ldbias": "0",
        "ldc": "N * B",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


@registry.reg("cuda.perm102_bmm_rrr_bias.config")
def gemm_rrr_config(func_attrs, dtype="float16"):
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
    return perm102_bmm_rrr.gemm_rrr_config(func_attrs, dtype)


@registry.reg("cuda.perm102_bmm_rrr_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=["M", "B", "K"], b_dims=["B", "K", "N"], c_dims=["M", "B", "N"]
    )

    mm_info = _get_problem_info(alpha_value=func_attrs.get("alpha", 1))
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(mm_info=mm_info)

    bmm_common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common_bias.SRC_TEMPLATE,
        problem_args,
        args_parser,
        bias_ptr_arg="memory_pool->RequestHalfTensorByIdx(3)",
    )


@registry.reg("cuda.perm102_bmm_rrr_bias.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    mm_info = _get_problem_info(alpha_value=func_attrs.get("alpha", 1))
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(mm_info=mm_info)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)

    return common.gen_function(
        func_attrs,
        common_bias.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.perm102_bmm_rrr_bias.func_decl")
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
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name, input_ndims=input_ndims, weight_ndims=weight_ndims
    )


@registry.reg("cuda.perm102_bmm_rrr_bias.func_call")
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
    bias = func_attrs["inputs"][2]
    return bmm_common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.perm102_bmm_rrr_bias.filter")
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
