# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from ... import registry
from . import common, common_bias_activation

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    split_k,
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
    (void*) a_ptr,
    (void*) b_ptr,
    (void*) bias_ptr,
    (void*) (c_ptr + output_offset),
    M * K,
    N * K,
    N,
    M * N,
    K,
    K,
    0,
    output_stride
"""
)


@registry.reg("cuda.gemm_rcr_bias_swish.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    return common_bias_activation.gemm_rcr_config(func_attrs, dtype)


@registry.reg("cuda.gemm_rcr_bias_swish.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    return common_bias_activation.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        PROBLEM_ARGS_TEMPLATE,
    )


@registry.reg("cuda.gemm_rcr_bias_swish.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    return common_bias_activation.gen_function(
        func_attrs,
        PROBLEM_ARGS_TEMPLATE,
        exec_cond_template,
        dim_info_dict,
    )


@registry.reg("cuda.gemm_rcr_bias_swish.func_decl")
def gen_function_decl(func_attrs):
    return common_bias_activation.gen_function_decl(func_attrs)


@registry.reg("cuda.gemm_rcr_bias_swish.func_call")
def gen_function_call(func_attrs, indent="  "):
    return common_bias_activation.gen_function_call(func_attrs, indent)


@registry.reg("cuda.gemm_rcr_bias_swish.filter")
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
