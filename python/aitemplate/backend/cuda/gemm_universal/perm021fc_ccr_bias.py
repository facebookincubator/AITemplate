# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[b, m, n] = bmm([b, k, m], [1, n, k])
in torch it is
# _3306 = _3305.permute(0, 2, 1)  # Transpose
# _3307 = _3306  # torch.reshape(_3306, (-1, 745))  # Reshape
# _3308 = torch.nn.functional.linear(_3307, self._1184, bias=self._1185)  # FC
"""
from ... import registry
from . import bmm_common, common, common_bias, perm021fc_ccr

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_problem_info(**kwargs):
    problem_args = {
        "beta_value": 1,
        "bias_ptr": "bias_ptr",
        "a_batch_stride": "M * K",
        "b_batch_stride": "0",
        "bias_batch_stride": "0",
        "c_batch_stride": "M * N",
        "lda": "M",
        "ldb": "K",
        "ldbias": "0",
        "ldc": "N",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


@registry.reg("cuda.perm021fc_ccr_bias.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
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
    return perm021fc_ccr.gemm_ccr_config(func_attrs, dtype)


@registry.reg("cuda.perm021fc_ccr_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=["B", "K", "M"], b_dims=["1", "N", "K"], c_dims=["B", "M", "N"]
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


@registry.reg("cuda.perm021fc_ccr_bias.gen_function")
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


@registry.reg("cuda.perm021fc_ccr_bias.func_decl")
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


@registry.reg("cuda.perm021fc_ccr_bias.func_call")
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


@registry.reg("cuda.perm021fc_ccr_bias.filter")
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
