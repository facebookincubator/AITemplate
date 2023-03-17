#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Common template for bmm
"""
import jinja2

from aitemplate.backend.rocm.gemm import common

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}ck::index_t stride_a = *a_dim2;
{{indent}}ck::index_t stride_b = *b_dim2;
{{indent}}ck::index_t stride_c = *c_dim2;

{{indent}}ck::index_t batch_stride_a = (*a_dim1) * stride_a;
{{indent}}ck::index_t batch_stride_b = (*b_dim1) * stride_b;
{{indent}}ck::index_t batch_stride_c = (*c_dim1) * stride_c;
"""
)

INPUT_ADDR_CALCULATOR = jinja2.Template(
    """
    {% if accessor_a.is_from_strided_tensor %}
      batch_stride_a = {{accessor_a.stride(0)}};
      stride_a = {{accessor_a.stride(1)}};
      offset_a = {{accessor_a.offset}}; // default to 0
    {% endif %}
    {% if accessor_b.is_from_strided_tensor %}
      batch_stride_b = {{accessor_b.stride(0)}};
      stride_b = {{accessor_b.stride(1)}};
      offset_b = {{accessor_b.offset}}; // default to 0
    {% endif %}
    """
)

# pylint: disable=C0103,C0415,W0611,C0301
OUTPUT_ADDR_CALCULATOR = jinja2.Template(
    """
  {% if output_accessor.is_from_strided_tensor %}
    batch_stride_c = {{output_accessor.stride(0)}};
    stride_c = {{output_accessor.stride(1)}};
    offset_c = {{output_accessor.offset}};
  {% endif %}
    """
)

EXTRA_HEADER_TEMPLATE = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp"
"""
)

EXTRA_HEADER_TEMPLATE_MULTI_D = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
"""
)

PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr) + offset_a,
{{indent}}                                static_cast<ck::half_t *>(weight_ptr) + offset_b,
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr) + offset_c,
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% if "bias" in gemm_flag %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% endif %}
{{indent}}                                stride_c,
{{indent}}                                batch_stride_a,
{{indent}}                                batch_stride_b,
{{indent}}                                batch_stride_c,
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

PROBLEM_ARGS_TEMPLATE_MULTI_D = jinja2.Template(
    """
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{% if "bias" in gemm_flag or gemm_flag == "add" %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% else %}
{{indent}}                                {},
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                B,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{% if gemm_flag == "add" %}
{{indent}}                                std::array<ck::index_t, 1>{stride_c},
{% elif gemm_flag == "bias" %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% else %}
{{indent}}                                {},
{% endif %}
{{indent}}                                stride_c,
{{indent}}                                batch_stride_a,
{{indent}}                                batch_stride_b,
{% if gemm_flag == "add" %}
{{indent}}                                std::array<ck::index_t, 1>{batch_stride_c},
{% elif gemm_flag == "bias" %}
{{indent}}                                std::array<ck::index_t, 1>{stride_c},
{% else %}
{{indent}}                                {},
{% endif %}
{{indent}}                                batch_stride_c,
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag in ["bias", "add"] %}
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
{% if "add" == gemm_flag %}
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // b: index 3
{% endif %}
{% if "bias" in gemm_flag %}
  memory_pool->AllocateHalfTensor(B*N, mem_pool_sz);  // b: index 3
{% endif %}
"""
)


# FIXME: consider to move this into a common function shared by both cuda
# and rocm
def reverse_dim_info_mapping(dim_info_dict, source, tensor_idx):
    def _fill(arr, idx, val):
        if len(arr) <= idx:
            arr = arr + [None] * (idx - len(arr) + 1)
        arr[idx] = val
        return arr

    ret = []
    for name, dim_infos in dim_info_dict.items():
        for dim_info in dim_infos:
            if dim_info.source == source and dim_info.tensor_idx == tensor_idx:
                for dim_idx in dim_info.dim_idx:
                    ret = _fill(ret, dim_idx, name)

    if None in ret:
        raise RuntimeError(
            "dim_info_dict for source: {}, tensor_idx: {} not complete.".format(
                source, tensor_idx
            )
        )

    return ret


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    args_parse,
    gemm_flag,
    problem_args_template=None,
    extra_header_template=None,
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
    if problem_args_template is None:
        if gemm_flag == "":
            problem_args_template = PROBLEM_ARGS_TEMPLATE
        else:
            problem_args_template = PROBLEM_ARGS_TEMPLATE_MULTI_D

    if extra_header_template is None:
        if gemm_flag == "":
            extra_header_template = EXTRA_HEADER_TEMPLATE
        else:
            extra_header_template = EXTRA_HEADER_TEMPLATE_MULTI_D

    return common.gen_profiler(
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
    problem_args_template=None,
    extra_header_template=None,
    extra_shape_template=EXTRA_SHAPE_TEMPLATE,
    extra_code="",
    input_addr_calculator="",
    output_addr_calculator="",
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
    input_addr_calculator : str
        Used to adjust input address based on input tensor accessors if accessors exist
    output_addr_calculator : str
        Used to adjust output address based on output tensor accessors if accessors exist


    Returns
    -------
    str
        The rendered template of generated function body.
    """
    if problem_args_template is None:
        if gemm_flag == "":
            problem_args_template = PROBLEM_ARGS_TEMPLATE
        else:
            problem_args_template = PROBLEM_ARGS_TEMPLATE_MULTI_D

    if extra_header_template is None:
        if gemm_flag == "":
            extra_header_template = EXTRA_HEADER_TEMPLATE
        else:
            extra_header_template = EXTRA_HEADER_TEMPLATE_MULTI_D

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
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
    )


def gen_function_decl(func_name, gemm_flag, pdims=0):
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
    return common.gen_function_decl(
        func_name=func_name, gemm_flag=gemm_flag, ndims=3, pdims=pdims
    )


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
