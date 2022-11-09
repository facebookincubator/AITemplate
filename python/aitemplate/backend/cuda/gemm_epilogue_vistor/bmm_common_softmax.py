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
Common functions and templates for bmm-family ops
"""
import jinja2

from ...common import gemm_common
from ..gemm_universal import common

from . import common_softmax

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  cutlass::half_t*,
  cutlass::half_t*,
{% if has_bias %}
  cutlass::half_t*,
{% endif %}
  cutlass::half_t*,
  cutlass::half_t*,
  float*,
  cutlass::half_t*,
  uint8_t*,
{% if support_split_k %}
  int,
{% endif %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
  cudaStream_t
);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{% if has_bias %}
{{indent}}    {{bias_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{{indent}}    {{d_ptr}},
{{indent}}    {{n_ptr}},
{{indent}}    {{soft_ptr}},
{{indent}}    global_workspace_,
{{indent}}    {{a_dim0_ptr}},
{{indent}}    {{a_dim1_ptr}},
{{indent}}    {{a_dim2_ptr}},
{{indent}}    {{b_dim0_ptr}},
{{indent}}    {{b_dim1_ptr}},
{{indent}}    {{b_dim2_ptr}},
{{indent}}    {{c_dim0_ptr}},
{{indent}}    {{c_dim1_ptr}},
{{indent}}    {{c_dim2_ptr}},
{{indent}}    stream
{{indent}});
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  // cast to int64_t to avoid overflow
  int64_t a_ptr_sz = static_cast<int64_t>(a_dim0) * static_cast<int64_t>(a_dim1) * static_cast<int64_t>(a_dim2);
  int64_t b_ptr_sz = static_cast<int64_t>(b_dim0) * static_cast<int64_t>(b_dim1) * static_cast<int64_t>(b_dim2);
  int64_t c_ptr_sz = static_cast<int64_t>(c_dim0) * static_cast<int64_t>(c_dim1) * static_cast<int64_t>(c_dim2);
  int64_t ptr_max_sz = std::max({a_ptr_sz, b_ptr_sz, c_ptr_sz});
  // TODO: special pool size for A100 L2 cache 40M
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 25) / ptr_max_sz)));


  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // c_ptr: index 2
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d_ptr: index 3
  memory_pool->AllocateFloatTensor(c_dim0 * c_dim1,  mem_pool_sz);  // n_ptr: index 4
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // soft_ptr: index 5
"""
)


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    src_template,
    problem_args_template,
    args_parser_template,
    emit_kernel=False,
    bias_ptr_arg=None,
):
    """Generate code for profiling"""
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    has_d = False
    if "has_d" in func_attrs:
        has_d = func_attrs["has_d"]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    file_pairs = []
    has_bias = bias_ptr_arg is not None
    assert not (has_d and has_bias)
    for op_name, op in op_instance.items():
        config = common_softmax.emit_instance(op, emit_kernel=emit_kernel)
        config_name = common.extract_config_name(config)
        name = "GemmInstance"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=name, config=config
        )
        exec_program = common_softmax.EXEC_TEMPLATE.render(
            indent="  ",
            instance=name,
            is_profiler=True,
            problem_args=problem_args_template.render(),
        )
        op_func = src_template.render(
            custom_libs=common_softmax.gen_custom_libs(),
            instances=instance,
            function_name="bmm",
            input_ndims=3,
            weight_ndims=3,
            shape_eval=shape_func,
            exec_paths=exec_program,
            has_d=has_d,
        )
        func_call = FUNC_CALL_TEMPLATE.render(
            func_name="bmm",
            a_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(0)",
            b_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(1)",
            has_bias=has_bias,
            bias_ptr=bias_ptr_arg,
            c_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(2)",
            d_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(3)",
            n_ptr="memory_pool->RequestTensorByIdx<float>(4)",
            soft_ptr="memory_pool->RequestTensorByIdx<cutlass::half_t>(5)",
            has_d=has_d,
            a_dim0_ptr="&a_dim0",
            a_dim1_ptr="&a_dim1",
            a_dim2_ptr="&a_dim2",
            b_dim0_ptr="&b_dim0",
            b_dim1_ptr="&b_dim1",
            b_dim2_ptr="&b_dim2",
            c_dim0_ptr="&c_dim0",
            c_dim1_ptr="&c_dim1",
            c_dim2_ptr="&c_dim2",
        )
        code = common_softmax.PROFILER_TEMPLATE.render(
            op_func=op_func,
            args_parse=args_parser_template.render(),
            func_call=func_call,
            name=name,
            tensor_decl=TENSOR_DECL_TEMPLATE.render(
                name=name, has_d=has_d, has_bias=has_bias
            ),
        )
        common.add_profiler(file_pairs, workdir, op_type, op_name, code)
    # build
    return common.build_profiler(file_pairs)


def gen_function_decl(func_attrs):
    """Rendering argument to function declaration template"""
    func_name = func_attrs["name"]
    has_d = False
    if "has_d" in func_attrs:
        has_d = func_attrs["has_d"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name, ndims=3, has_d=has_d)


def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    problem_args,
):
    """Generate the code for main function"""
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_softmax.gen_function(
        func_attrs,
        common_softmax.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        dim_info_dict=dim_info_dict,
        emit_kernel=True,
    )


def gen_function_call(func_attrs, indent="  ", bias_ptr_arg=None):
    """Rendering the code to function call template"""

    a = func_attrs["inputs"][0]
    ashape = func_attrs["input_accessors"][0].original_shapes
    b = func_attrs["inputs"][1]
    bshape = func_attrs["input_accessors"][1].original_shapes

    c = func_attrs["inputs"][2]
    d = func_attrs["inputs"][3]
    n = func_attrs["inputs"][4]

    soft = func_attrs["outputs"][0]
    cshape = func_attrs["output_accessors"][0].original_shapes
    has_d = False
    has_bias = bias_ptr_arg is not None
    assert not (has_d and has_bias)
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        has_bias=has_bias,
        bias_ptr=bias_ptr_arg,
        c_ptr=c._attrs["name"],
        d_ptr=d._attrs["name"],
        n_ptr=n._attrs["name"],
        soft_ptr=soft._attrs["name"],
        has_d=has_d,
        a_dim0_ptr="&" + ashape[0]._attrs["name"],
        a_dim1_ptr="&" + ashape[1]._attrs["name"],
        a_dim2_ptr="&" + ashape[2]._attrs["name"],
        b_dim0_ptr="&" + bshape[0]._attrs["name"],
        b_dim1_ptr="&" + bshape[1]._attrs["name"],
        b_dim2_ptr="&" + bshape[2]._attrs["name"],
        c_dim0_ptr="&" + cshape[0]._attrs["name"],
        c_dim1_ptr="&" + cshape[1]._attrs["name"],
        c_dim2_ptr="&" + cshape[2]._attrs["name"],
        indent=indent,
    )
