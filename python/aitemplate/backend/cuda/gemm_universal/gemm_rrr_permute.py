# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for A[RowMajor], B[ColMajor], C[RowMajor]
This is special in template based gemm solution
This is used for `torch.nn.functional.linear`
When use for `linear`, need set A->Data, B->Weight
"""
import jinja2

from ... import registry
from ..gemm_universal import common
from . import common_permute

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);

  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = K;
  int64_t b_dim1 = N;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
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
    N,
    N,
    output_stride,
"""
)


@registry.reg("cuda.gemm_rrr_permute.config")
def gemm_rrr_permute_config(func_attrs, dtype="float16"):
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

        return common_permute.default_fproc_f16(
            op=op,
            a_layout=cutlass_lib.library.LayoutType.RowMajor,
            b_layout=cutlass_lib.library.LayoutType.RowMajor,
            c_layout=cutlass_lib.library.LayoutType.RowMajor,
            epiligue_name=func_attrs["epilogue"],
            permute_layout=func_attrs["layout"],
        )

    func_attrs["op_instance"] = common_permute.extract_config(fproc_f16, func_attrs)


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
        stride_dim="N"
    )
    common_permute.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        src_template,
        problem_args_template,
        ARGS_PARSER_TEMPLATE,
        emit_kernel=True,
        support_split_k=True,
        output_addr_calculator=output_addr_calculator,
        bias_ptr_arg=bias_ptr_arg,
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_rrr_permute.gen_profiler")
def gen_profiler(func_attrs, workdir, dim_info_dict):
    return common_gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        common.SRC_TEMPLATE,
        PROBLEM_ARGS_TEMPLATE,
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rrr_permute.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args_template=None,
):
    if problem_args_template is None:
        problem_args = PROBLEM_ARGS_TEMPLATE.render()
    else:
        problem_args = problem_args_template.render()

    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    return common_permute.gen_function(
        func_attrs,
        common.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims,
        weight_ndims,
        output_ndims,
        dim_info_dict,
        emit_kernel=True,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="*b_dim1", output_accessor=func_attrs["output_accessors"][0]
        ),
        extra_code=common_permute.EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rrr_permute.func_decl")
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
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=True,
    )


@registry.reg("cuda.gemm_rrr_permute.func_call")
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
    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]

    output = func_attrs["outputs"][0]
    has_bias = False
    adims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][0].original_shapes
    ]
    bdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["input_accessors"][1].original_shapes
    ]
    cdims = [
        "&" + dim._attrs["name"]
        for dim in func_attrs["output_accessors"][0].original_shapes
    ]
    return common.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        has_bias=has_bias,
        c_ptr=output._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )


@registry.reg("cuda.gemm_rrr_permute.filter")
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
