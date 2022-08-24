# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
GEMM Specialization for
C = UnaryOp2(BinaryOp2(BinaryOp1(UnaryOp1(GeMM(A, B) + bias), D1), D2)),
"""

import re
from functools import partial

import jinja2

from ...common import gemm_common
from ...target import Target

from . import common, gemm_rcr

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


# For config extraction.
GEMM_UNIVERSAL_WITH_BROADCAST_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        cutlass::half_t, {{layout.cutlass_layout_a}},
        cutlass::half_t, {{layout.cutlass_layout_b}},
        cutlass::half_t, {{layout.cutlass_layout_c}},
        {{acc_type}},
        cutlass::arch::OpClassTensorOp,
        {{arch}},
        {{tb_shape}},
        {{warp_shape}},
        {{instruction_shape}},
        {{epilogue_functor}}<
            cutlass::half_t, {{acc_type}}, {{acc_type}},
            cutlass::half_t, {{epilogue_vector_length}},
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
    cutlass::gemm::GemmUniversalMode::kGemm,
    { {{layout.m}}, {{layout.n}}, {{layout.k}} },
{% if support_split_k %}
    split_k,
{% else %}
    1,
{% endif %}
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
    (void*) (a_ptr + input_a_offset),
    (void*) (b_ptr + input_b_offset),
    (void*) d0_ptr,
{% if has_d1 %}
    (void*) d1_ptr,
{% else %}
    nullptr,
{% endif %}
    (void*) (c_ptr + output_offset),
    (void*) bias_ptr,
    nullptr,
    /*batch_stride_A*/ input_a_batch_stride,
    /*batch_stride_B*/ input_b_batch_stride,
    /*batch_stride_C1*/ 0,
    /*batch_stride_C2*/ 0,
    /*batch_stride_D*/ 0,
    /*batch_stride_Vector*/ 0,
    /*batch_stride_Tensor*/ 0,
    input_a_stride,
    input_b_stride,
    {{layout.stride_c}},
{% if has_d1 %}
    {{layout.stride_c}},
{% else %}
    0,
{% endif %}
    output_stride,
    /*ldr*/ 0,
    /*/ldt*/ 0
"""
)

# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,
    { {{layout.m}}, {{layout.n}}, {{layout.k}} },
{% if support_split_k %}
    split_k,
{% else %}
    1,
{% endif %}
    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
    (void*) a_ptr,
    (void*) b_ptr,
    (void*) d0_ptr,
{% if has_d1 %}
    (void*) d1_ptr,
{% else %}
    nullptr,
{% endif %}
    (void*) (c_ptr + output_offset),
    (void*) bias_ptr,
    nullptr,
    /*batch_stride_A*/ 0,
    /*batch_stride_B*/ 0,
    /*batch_stride_C1*/ 0,
    /*batch_stride_C2*/ 0,
    /*batch_stride_D*/ 0,
    /*batch_stride_Vector*/ 0,
    /*batch_stride_Tensor*/ 0,
    {{layout.stride_a}},
    {{layout.stride_b}},
    {{layout.stride_c}},
{% if has_d1 %}
    {{layout.stride_c}},
{% else %}
    0,
{% endif %}
    output_stride,
    /*ldr*/ 0,
    /*/ldt*/ 0
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block_v2.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"


#define CUTLASS_CHECK(status)                                                                    \\
  {                                                                                              \\
    cutlass::Status error = status;                                                              \\
    if (error != cutlass::Status::kSuccess) {                                                    \\
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __FILE__ \\
                << ":" << __LINE__ << std::endl;                                                 \\
      exit(EXIT_FAILURE);                                                                        \\
    }                                                                                            \\
  }

{{instances}}

void {{function_name}} (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
    cutlass::half_t* bias_ptr,
    cutlass::half_t* d0_ptr,
{% if has_d1 %}
    cutlass::half_t* d1_ptr,
{% endif %}
    cutlass::half_t* c_ptr,
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
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
{% if has_d1 %}
  cutlass::half_t*,
{% endif %}
  cutlass::half_t*,
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
{{indent}}    {{a_ptr}},
{{indent}}    {{b_ptr}},
{{indent}}    {{bias_ptr}},
{{indent}}    {{d0_ptr}},
{% if has_d1 %}
{{indent}}    {{d1_ptr}},
{% endif %}
{{indent}}    {{c_ptr}},
{{indent}}    global_workspace,
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
  // TODO: special pool size for A100 L2 cache 40M
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 25) / ptr_max_sz)));

  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // c_ptr: index 2
  memory_pool->AllocateHalfTensor(c_dim1, mem_pool_sz);  // bias_ptr: index 3
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d0 ptr: index 4
{% if has_d1 %}
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d1 ptr: index 5
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
    """[summary]

    Parameters
    ----------
    func_attrs : [type]
        [description]
    layout : [type]
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
    common.make_fproc_f16(func_attrs, layout)


def gen_profiler(
    func_attrs,
    workdir,
    dim_info_dict,
    layout,
    unary_op1,
    binary_op1,
    binary_op2,
    unary_op2,
):
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

    file_pairs = []
    for op_name, op in op_instance.items():
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
            ),
        )
        config_name = common.extract_config_name(config)
        name = "GemmInstance"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=name, config=config
        )
        exec_program = common.EXEC_TEMPLATE.render(
            indent="  ",
            instance=name,
            is_profiler=True,
            problem_args=PROFILER_PROBLEM_ARGS_TEMPLATE.render(
                support_split_k=support_split_k, layout=layout, has_d1=has_d1
            ),
        )
        input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
            input_ndims=ndims,
            weight_ndims=ndims,
            output_ndims=ndims,
        )
        op_func = SRC_TEMPLATE.render(
            instances=instance,
            function_name="gemm",
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
        func_call = FUNC_CALL_TEMPLATE.render(
            func_name="gemm",
            a_ptr="memory_pool->RequestHalfTensorByIdx(0)",
            b_ptr="memory_pool->RequestHalfTensorByIdx(1)",
            c_ptr="memory_pool->RequestHalfTensorByIdx(2)",
            d0_ptr="memory_pool->RequestHalfTensorByIdx(4)",
            d1_ptr="memory_pool->RequestHalfTensorByIdx(5)",
            bias_ptr="memory_pool->RequestHalfTensorByIdx(3)",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
            support_split_k=support_split_k,
            split_k="split_k",
            has_d1=has_d1,
        )
        code = common.PROFILER_TEMPLATE.render(
            op_func=op_func,
            args_parse=ARGS_PARSER_TEMPLATE.render(
                layout=layout, support_split_k=support_split_k
            ),
            func_call=func_call,
            name=name,
            tensor_decl=TENSOR_DECL_TEMPLATE.render(name=name, has_d1=has_d1),
        )
        common.add_profiler(file_pairs, workdir, op_type, op_name, code)
    # build
    common.build_profiler(file_pairs)


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
    input_addr_calculator = gemm_rcr.get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    support_split_k = _support_split_k(func_attrs)
    has_d1 = common.has_d1(func_attrs)
    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        layout=layout, support_split_k=support_split_k, has_d1=has_d1
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
        ),
        support_split_k=support_split_k,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N",
            output_accessor=func_attrs["output_accessors"][0],
        ),
    )


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
