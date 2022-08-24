# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from ... import registry
from . import common, common_bias, gemm_rcr

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


# used for real execution
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    split_k,
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
    (void*) (a_ptr + input_a_offset),
    (void*) (b_ptr + input_b_offset),
    (void*) bias_ptr,
    (void*) (c_ptr + output_offset),
    input_a_batch_stride,
    input_b_batch_stride,
    /*bias_batch_stride*/ N,
    /*output_batch_stride*/ M * N,
    input_a_stride,
    input_b_stride,
    /*bias_stride*/ 0,
    output_stride
"""
)


# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
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


@registry.reg("cuda.gemm_rcr_bias.config")
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
    return gemm_rcr.gemm_rcr_config(func_attrs, dtype)


@registry.reg("cuda.gemm_rcr_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    gemm_rcr.common_gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common_bias.SRC_TEMPLATE,
        PROFILER_PROBLEM_ARGS_TEMPLATE,
        bias_ptr_arg="memory_pool->RequestHalfTensorByIdx(3)",
    )


@registry.reg("cuda.gemm_rcr_bias.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_addr_calculator = gemm_rcr.get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    problem_args = PROBLEM_ARGS_TEMPLATE.render()
    return common.gen_function(
        func_attrs,
        common_bias.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims,
        weight_ndims,
        output_ndims,
        dim_info_dict,
        support_split_k=True,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
    )


@registry.reg("cuda.gemm_rcr_bias.func_decl")
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
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["inputs"][0]._attrs["shape"])
    weight_ndims = len(func_attrs["inputs"][1]._attrs["shape"])
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("cuda.gemm_rcr_bias.func_call")
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
    return common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.gemm_rcr_bias.filter")
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
