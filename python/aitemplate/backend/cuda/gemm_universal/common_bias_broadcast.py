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
GEMM Specialization for
C = UnaryOp2(BinaryOp2(BinaryOp1(UnaryOp1(GeMM(A, B) + bias), D1), D2)),
"""

import re
from functools import partial

import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import gemm_common

from aitemplate.backend.cuda.gemm_universal import common, gemm_rcr
from aitemplate.backend.target import Target

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


# For config extraction.
GEMM_UNIVERSAL_WITH_BROADCAST_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        {{elem_type}}, {{layout.cutlass_layout_a}},
        {{elem_type}}, {{layout.cutlass_layout_b}},
        {{elem_type}}, {{layout.cutlass_layout_c}},
        {{acc_type}},
        cutlass::arch::OpClassTensorOp,
        {{arch}},
        {{tb_shape}},
        {{warp_shape}},
        {{instruction_shape}},
        {{epilogue_functor}}<
            {{elem_type}}, {{acc_type}}, {{acc_type}},
            {{elem_type}}, {{epilogue_vector_length}},
            {{unary_op1}}, {{binary_op1}}, {{unary_op2}}
{% if has_d1 %}
            , {{binary_op2}}
{% endif %}
        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        {{stage}},
        {{alignment_a}},
        {{alignment_b}}
    >;
"""
)

# For func codegen.
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    { {{layout.m}}, {{layout.n}}, {{layout.k}} },            // GemmCoord problem_size
{% if support_split_k %}
    split_k,                                                 // int batch_count
{% else %}
    1,                                                       // int batch_count
{% endif %}
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,          // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,          // void const * ptr_B
    ({{elem_output_type}}*)(d0_ptr),                         // void const * ptr_C1
{% if has_d1 %}
    ({{elem_output_type}}*)(d1_ptr),                         // void const * ptr_C2
{% endif %}
    ({{elem_output_type}}*) (c_ptr) + output_offset,         // void * ptr_D
    ({{elem_input_type}}*) (bias_ptr),                       // void * ptr_Vector
    nullptr,                                                 // void * ptr_Tensor
    input_a_batch_stride,                                    // int64_t batch_stride_A
    input_b_batch_stride,                                    // int64_t batch_stride_B
    0,                                                       // int64_t batch_stride_C1
{% if has_d1 %}
    0,                                                       // int64_t batch_stride_C2
{% endif %}
    0,                                                       // int64_t batch_stride_D
    0,                                                       // int64_t batch_stride_Vector
    0,                                                       // int64_t batch_stride_Tensor
    input_a_stride,                                          // typename LayoutA::Stride::Index lda
    input_b_stride,                                          // typename LayoutB::Stride::Index ldb
    {{layout.stride_c}},                                     // typename LayoutC::Stride::Index ldc1
{% if has_d1 %}
    {{layout.stride_c}},                                     // typename LayoutC::Stride::Index ldc2
{% endif %}
    output_stride,                                           // typename LayoutC::Stride::Index ldd
    0,                                                       // typename LayoutC::Stride::Index ldr
    0,                                                       // typename LayoutC::Stride::Index ldt
"""
)

# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    { {{layout.m}}, {{layout.n}}, {{layout.k}} },            // GemmCoord problem_size
{% if support_split_k %}
    split_k,                                                 // int batch_count
{% else %}
    1,                                                       // int batch_count
{% endif %}
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*) a_ptr,                            // void const * ptr_A
    ({{elem_input_type}}*) b_ptr,                            // void const * ptr_B
    ({{elem_output_type}}*) d0_ptr,                          // void const * ptr_C1
{% if has_d1 %}
    ({{elem_output_type}}*) d1_ptr,                          // void const * ptr_C2
{% endif %}
    ({{elem_output_type}}*) (c_ptr) + output_offset,         // void * ptr_D
    ({{elem_input_type}}*) bias_ptr,                         // void * ptr_Vector
    nullptr,                                                 // void * ptr_Tensor
    0,                                                       // int64_t batch_stride_A
    0,                                                       // int64_t batch_stride_B
    0,                                                       // int64_t batch_stride_C1
{% if has_d1 %}
    0,                                                       // int64_t batch_stride_C2
{% endif %}
    0,                                                       // int64_t batch_stride_D
    0,                                                       // int64_t batch_stride_Vector
    0,                                                       // int64_t batch_stride_Tensor
    {{layout.stride_a}},                                     // typename LayoutA::Stride::Index lda
    {{layout.stride_b}},                                     // typename LayoutA::Stride::Index ldb
    {{layout.stride_c}},                                     // typename LayoutA::Stride::Index ldc1
{% if has_d1 %}
    {{layout.stride_c}},                                     // typename LayoutA::Stride::Index ldc2
{% endif %}
    output_stride,                                           // typename LayoutA::Stride::Index ldd
    0,                                                       // typename LayoutA::Stride::Index ldr
    0,                                                       // typename LayoutA::Stride::Index ldt
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

using bfloat16 = nv_bfloat16;

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \\
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \\
      std::cerr << msg << std::endl;                                                  \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}

{% if is_profiler %}
template <typename GemmInstance>
void {{function_name}} (
    GemmInstance& gemm_op,
{% else %}
void {{function_name}} (
{% endif %}
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* d0_ptr,
{% if has_d1 %}
    void* d1_ptr,
{% endif %}
    void* c_ptr,
    uint8_t* workspace,
{% if support_split_k %}
    int split_k,
{% endif %}
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
    cudaStream_t stream
  ) {
  {{shape_eval}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{extra_shape}}
  {{input_output_checks}}

  if (!bias_ptr) {
    throw std::runtime_error("bias is null!");
  }
  if (!d0_ptr) {
    throw std::runtime_error("d0_ptr is null!");
  }
{% if has_d1 %}
  if (!d1_ptr) {
    throw std::runtime_error("d1_ptr is null!");
  }
{% endif %}

  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this {{function_name}} specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

# For function declaration codegen.
FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  void*,
{% if has_d1 %}
  void*,
{% endif %}
  void*,
  uint8_t*,
{% if support_split_k %}
    int,
{% endif %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(weight_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
  cudaStream_t
);
"""
)


# For function call codegen.
FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{% if is_profiler %}
{{indent}}    gemm_op,
{% endif %}
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{{indent}}    {{bias_ptr}},
{{indent}}    {{d0_ptr}},
{% if has_d1 %}
{{indent}}    {{d1_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{{indent}}    global_workspace_,
{% if support_split_k %}
{{indent}} {{split_k}},
{% endif %}
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
{{indent}}}
"""
)

# For profiler codegen.
ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
{% if support_split_k %}
  int split_k = std::atoi(argv[4]);
{% endif %}
  {{layout.args_parser}}
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = a_dim0 * a_dim1;
  int64_t b_ptr_sz = b_dim0 * b_dim1;
  int64_t c_ptr_sz = c_dim0 * c_dim1;
  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});
  size_t one_copy_sz = a_ptr_sz + b_ptr_sz + c_ptr_sz + c_dim1 + c_ptr_sz;
{% if has_d1 %}
  one_copy_sz += c_ptr_sz;
{%endif%}
  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2
  memory_pool->AllocateTensor(c_dim1, mem_pool_sz);  // bias_ptr: index 3
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz);  // d0 ptr: index 4
{% if has_d1 %}
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz);  // d1 ptr: index 5
{% endif %}
"""
)


def _support_split_k(func_attrs):
    return func_attrs["split_k"] is not None


def gemm_bias_broadcast_instance(
    op_def,
    func_attrs,
    for_profiler,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
    elem_type,
):
    """
    adjust gemm instance with respect to input_accessors, layout and epilogue ops
    """
    op_def = common.update_alignments_in_gemm_instance(op_def, func_attrs, for_profiler)
    gemm_universal_params = common.get_gemm_instance_template_params(op_def)
    epilogue_pattern = re.compile(r"\s*(cutlass::epilogue::thread::.*)\s*<")
    match = epilogue_pattern.match(gemm_universal_params[9])
    if match is None:
        raise RuntimeError("Invalid epilogue functor:\n" + gemm_universal_params[9])
    epilogue_functor = match.groups()[0]

    if (
        "use_fp16_acc" in Target.current()._kwargs
        and Target.current()._kwargs["use_fp16_acc"]
        and elem_type == "cutlass::half_t"
    ):
        acc_type = "cutlass::half_t"
    else:
        acc_type = "float"
    gemm_universal_with_broadcast_params = (
        GEMM_UNIVERSAL_WITH_BROADCAST_TEMPLATE.render(
            arch=gemm_universal_params[5],
            tb_shape=gemm_universal_params[6],
            warp_shape=gemm_universal_params[7],
            instruction_shape=gemm_universal_params[8],
            epilogue_functor=epilogue_functor,
            epilogue_vector_length=gemm_universal_params[11],
            unary_op1=unary_op1,
            binary_op1=binary_op1,
            binary_op2=binary_op2,
            unary_op2=unary_op2,
            stage=gemm_universal_params[16],
            alignment_a=gemm_universal_params[17],
            alignment_b=gemm_universal_params[18],
            layout=layout,
            acc_type=acc_type,
            elem_type=elem_type,
            has_d1=(binary_op2 is not None),
        )
    )
    res = re.sub(
        r"cutlass::gemm::device::Gemm<[\s\S]+>;",
        gemm_universal_with_broadcast_params,
        op_def,
    )
    return res


def gemm_bias_broadcast_config(func_attrs, layout, dtype="float16"):
    common.make_fproc(func_attrs, layout)


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    op_type = func_attrs["op"]
    support_split_k = _support_split_k(func_attrs)
    op_instance = func_attrs["op_instance"]
    has_d1 = common.has_d1(func_attrs)

    ndims = 2
    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    instance_name_base = "GemmInstance"
    exec_program = common.EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        problem_args=PROFILER_PROBLEM_ARGS_TEMPLATE.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
            support_split_k=support_split_k,
            layout=layout,
            has_d1=has_d1,
        ),
    )
    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
    )

    function_name = "gemm"
    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = common.emit_instance(
            op,
            for_profiler=True,
            f_instance_convertor=partial(
                gemm_bias_broadcast_instance,
                layout=layout,
                unary_op1=unary_op1,
                binary_op1=binary_op1,
                binary_op2=binary_op2,
                unary_op2=unary_op2,
                elem_type=elem_input_type,
            ),
        )
        config_name = common.extract_config_name(config)
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_op_{instance_idx}"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=instance_name, config=config
        )
        benchmark_instance = common.BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
            support_split_k=support_split_k,
            split_k="split_k",
        )
        instances.append(instance)
        benchmark_instances.append(benchmark_instance)
    op_func = SRC_TEMPLATE.render(
        is_profiler=True,
        instances="\n".join(instances),
        function_name=function_name,
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        input_ndims=ndims,
        weight_ndims=ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_program,
        output_addr_calculator=common.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N"
        ),
        support_split_k=support_split_k,
        has_d1=has_d1,
    )
    benchmark_adims = ["a_dim" + str(i) for i in range(ndims)]
    benchmark_bdims = ["b_dim" + str(i) for i in range(ndims)]
    benchmark_cdims = ["c_dim" + str(i) for i in range(ndims)]
    func_call = FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name="gemm",
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b_ptr="memory_pool->RequestTensorByIdx(1)",
        c_ptr="memory_pool->RequestTensorByIdx(2)",
        d0_ptr="memory_pool->RequestTensorByIdx(4)",
        d1_ptr="memory_pool->RequestTensorByIdx(5)",
        bias_ptr="memory_pool->RequestTensorByIdx(3)",
        adims=benchmark_adims,
        bdims=benchmark_bdims,
        cdims=benchmark_cdims,
        support_split_k=support_split_k,
        split_k="split_k",
        has_d1=has_d1,
    )
    code = common.PROFILER_TEMPLATE.render(
        op_func=op_func,
        has_bias=True,
        has_d=True,
        has_d1=has_d1,
        support_split_k=support_split_k,
        args_parse=ARGS_PARSER_TEMPLATE.render(
            layout=layout, support_split_k=support_split_k
        ),
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        func_call=func_call,
        name=instance_name_base,
        tensor_decl=TENSOR_DECL_TEMPLATE.render(has_d1=has_d1),
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return common.build_profiler(file_pairs)


def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    input_addr_calculator = gemm_rcr.get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    support_split_k = _support_split_k(func_attrs)
    has_d1 = common.has_d1(func_attrs)
    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        layout=layout,
        support_split_k=support_split_k,
        has_d1=has_d1,
    )
    return common.gen_function(
        func_attrs,
        SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims,
        weight_ndims,
        output_ndims,
        dim_info_dict,
        f_instance_convertor=partial(
            gemm_bias_broadcast_instance,
            layout=layout,
            unary_op1=unary_op1,
            binary_op1=binary_op1,
            binary_op2=binary_op2,
            unary_op2=unary_op2,
            elem_type=elem_input_type,
        ),
        support_split_k=support_split_k,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N",
            output_accessor=func_attrs["output_accessors"][0],
        ),
    )


def gen_function_decl(func_attrs):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        support_split_k=_support_split_k(func_attrs),
        has_d1=common.has_d1(func_attrs),
    )


def gen_function_call(func_attrs, indent="  "):
    has_d1 = common.has_d1(func_attrs)
    if has_d1:
        (a, b, bias, d0, d1) = func_attrs["inputs"]
    else:
        (a, b, bias, d0) = func_attrs["inputs"]
        d1 = None
    c = func_attrs["outputs"][0]
    # overwrite the global defs if we have input TensorAccessor
    local_dim_defs = common.gen_local_dim_defs(func_attrs, indent=indent)
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
    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b_ptr=b._attrs["name"],
        bias_ptr=bias._attrs["name"],
        d0_ptr=d0._attrs["name"],
        d1_ptr=d1._attrs["name"] if has_d1 else "",
        c_ptr=c._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
        support_split_k=_support_split_k(func_attrs),
        has_d1=has_d1,
    )
