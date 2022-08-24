# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Common template for bmm
"""
import jinja2

from . import common

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}const int64_t stride_a = *a_dim2;
{{indent}}const int64_t stride_b = *b_dim2;
{{indent}}const int64_t stride_c = *c_dim2;
"""
)
EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp"
"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% endif %}
{{indent}}                                stride_c,
{{indent}}                                M*K,
{{indent}}                                N*K,
{{indent}}                                M*N,
{{indent}}                                B,
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "bias" %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif gemm_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif gemm_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% endif %}
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = B*M*K;
  int64_t b_ptr_sz = B*N*K;
  int64_t c_ptr_sz = B*M*N;
  int64_t ptr_max_sz = std::max({a_ptr_sz, b_ptr_sz, c_ptr_sz});
  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_max_sz)));

  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // x: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // w: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // y: index 2
{% if "bias" in gemm_flag %}
  memory_pool->AllocateHalfTensor(N, mem_pool_sz);  // b: index 3
{% endif %}
"""
)


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    args_parse,
    gemm_flag,
    problem_args_template=PROBLEM_ARGS_TEMPLATE,
    extra_header_template=EXTRA_HEADER_TEMPLATE,
    tensor_decl_template=TENSOR_DECL_TEMPLATE,
    extra_shape_template=EXTRA_SHAPE_TEMPLATE,
    extra_code="",
):
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from bmm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    args_parse: str
        Profiler input argument parser.
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu','bias_sigmoid','bias_add_relu'.
    extra_code : str
        Extra code for self-defined operators.
    """
    common.gen_profiler(
        func_attrs,
        workdir,
        dim_info_dict,
        args_parse,
        gemm_flag,
        extra_code=extra_code,
        ndims=3,
        extra_shape_template=extra_shape_template,
        problem_args_template=problem_args_template,
        extra_header_template=extra_header_template,
        tensor_decl_template=tensor_decl_template,
    )


def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    gemm_flag,
    problem_args_template=PROBLEM_ARGS_TEMPLATE,
    extra_header_template=EXTRA_HEADER_TEMPLATE,
    extra_shape_template=EXTRA_SHAPE_TEMPLATE,
    extra_code="",
):
    """Generate function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu','bias_add_relu'.
    extra_code : str
        Extra code for self-defined operators.


    Returns
    -------
    str
        The rendered template of generated function body.
    """
    return common.gen_function(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
        gemm_flag,
        extra_code=extra_code,
        ndims=3,
        extra_shape_template=extra_shape_template,
        problem_args_template=problem_args_template,
        extra_header_template=extra_header_template,
    )


def gen_function_decl(func_name, gemm_flag):
    """Generates function declarations.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu'.

    Returns
    -------
    str
        The rentered template of function declaration.
    """
    return common.gen_function_decl(func_name=func_name, gemm_flag=gemm_flag, ndims=3)


def gen_function_call(func_attrs, indent="  ", gemm_flag=""):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".
    gemm_flag : str
        Flag telling which backend should be generated. options are '','bias','bias_relu'.

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    return common.gen_function_call(func_attrs, indent=indent, gemm_flag=gemm_flag)
