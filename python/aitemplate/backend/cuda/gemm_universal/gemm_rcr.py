# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from ... import registry
from . import common
from .layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);

  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = N;
  int64_t b_dim1 = K;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""
)

# used for real execution
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    split_k,
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
    (void*) (a_ptr + input_a_offset),
    (void*) (b_ptr + input_b_offset),
    (void*) (c_ptr + output_offset),
    (void*) (c_ptr + output_offset),
    input_a_batch_stride,
    input_b_batch_stride,
    /*output_batch_stride*/ M * N,
    /*output_batch_stride*/ M * N,
    input_a_stride,
    input_b_stride,
    output_stride,
    output_stride
"""
)


# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    split_k,
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
    (void*) a_ptr,
    (void*) b_ptr,
    (void*) c_ptr,
    (void*) (c_ptr + output_offset),
    M * K,
    N * K,
    M * N,
    M * N,
    K,
    K,
    N,
    output_stride
"""
)


@registry.reg("cuda.gemm_rcr.config")
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
    common.make_fproc_f16(func_attrs, RCR)


def common_gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    src_template,
    problem_args_template,
    bias_ptr_arg=None,
    extra_code="",
):
    output_addr_calculator = common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        stride_dim="*b_dim0"
    )
    common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        src_template,
        problem_args_template,
        ARGS_PARSER_TEMPLATE,
        support_split_k=True,
        output_addr_calculator=output_addr_calculator,
        bias_ptr_arg=bias_ptr_arg,
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_rcr.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    return common_gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common.SRC_TEMPLATE,
        PROFILER_PROBLEM_ARGS_TEMPLATE,
    )


def get_input_addr_calculator(func_attrs):
    input_a_batch_stride_dim = "M * K"
    input_a_stride_k_dim = "K"
    input_a_offset = 0
    input_b_batch_stride_dim = "N * K"
    input_b_stride_k_dim = "K"
    input_b_offset = 0

    if "input_accessors" in func_attrs:
        input_a_accessor = func_attrs["input_accessors"][0]
        input_b_accessor = func_attrs["input_accessors"][1]
        if input_a_accessor.is_from_strided_tensor:
            input_a_offset = input_a_accessor.offset
            shapes = (
                input_a_accessor.actual_shapes
                if input_a_accessor.actual_shapes
                else input_a_accessor.original_shapes
            )
            input_a_stride_k_dim = input_a_accessor.stride(len(shapes) - 2)

        if input_b_accessor.is_from_strided_tensor:
            input_b_offset = input_b_accessor.offset
            shapes = (
                input_b_accessor.actual_shapes
                if input_b_accessor.actual_shapes
                else input_b_accessor.original_shapes
            )
            input_b_stride_k_dim = input_b_accessor.stride(len(shapes) - 2)

    input_addr_calculator = common.INPUT_ADDR_CALCULATOR.render(
        input_a_batch_stride_dim=input_a_batch_stride_dim,
        input_a_stride_dim=input_a_stride_k_dim,
        input_a_offset_val=input_a_offset,
        input_b_batch_stride_dim=input_b_batch_stride_dim,
        input_b_stride_dim=input_b_stride_k_dim,
        input_b_offset_val=input_b_offset,
    )
    return input_addr_calculator


@registry.reg("cuda.gemm_rcr.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_addr_calculator = get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    problem_args = PROBLEM_ARGS_TEMPLATE.render()
    return common.gen_function(
        func_attrs,
        common.SRC_TEMPLATE,
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


@registry.reg("cuda.gemm_rcr.func_decl")
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
    return common.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("cuda.gemm_rcr.func_call")
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
    return common.gen_function_call(func_attrs, indent)


@registry.reg("cuda.gemm_rcr.filter")
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
