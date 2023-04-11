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
Common codegen functions for dual gemm.
D0 = epilogue0(X @ B0, C0)
D1 = epilogue0(X @ B1, C1)
D2 = element_wise(D0, D1)
"""

from functools import partial
from hashlib import sha1
from typing import Any, Dict

import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import gemm_common
from aitemplate.backend.cuda.gemm_universal import common
from aitemplate.backend.target import Target

from aitemplate.utils import alignment


# pylint: disable=C0301,C0415,R1705

# HACK: we don't record different permutation shape,
# because it has little impact on execution time compared.
# Therefore, no matter what permutation shape it is,
# we will use the same kernel, i.e. the first generated perm_shape
# At runtime, the kernel will be regenerated and thus the correctness will not be affected.
KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass_{{opcode_class_name}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = 1;
{% for dim in adims %}
  a_ptr_sz *= {{dim}};
{% endfor %}

  int64_t b0_ptr_sz = 1;
{% for dim in bdims %}
  b0_ptr_sz *= {{dim}};
{% endfor %}

  int64_t b1_ptr_sz = b0_ptr_sz;
{% if broadcast_b1 %}
  // scale b1_ptr_sz down by the broadcasted dim
  b1_ptr_sz /= {{ bdims[broadcasted_bdim_id] }};
{% endif %}

  int64_t c_ptr_sz = 1;
{% for dim in cdims %}
  c_ptr_sz *= {{dim}};
{% endfor %}

  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b0_ptr_sz, c_ptr_sz});
  size_t one_copy_sz = a_ptr_sz + b0_ptr_sz + c_ptr_sz;
{% if has_bias %}
  one_copy_sz += b1_ptr_sz;
{%endif%}
  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b0_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2

{% if has_bias %}
  memory_pool->AllocateTensor(b1_ptr_sz, mem_pool_sz);  // b_ptr: index 3
{% endif %}

"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
//  TODO: cast to right dtype
//{{indent}}using ElementComputeEpilogue = typename {{instance}}::ElementAccumulator;
{{indent}}using ElementCompute = typename {{instance}}::DualGemmKernel::Epilogue0::OutputOp::ElementCompute;

{{indent}}using coord_t = cutlass::gemm::GemmCoord::Index;
{{indent}}typename {{instance}}::Arguments arguments{

{{problem_args}}

{{indent}}};
{% if is_profiler %}
{{indent}}size_t workspace_size = gemm_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);

{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% else %}
{{indent}}{{instance}} gemm_op;
{% endif %}

{{indent}} auto status = gemm_op.can_implement(arguments);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = gemm_op.initialize(arguments, workspace, stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = gemm_op(stream);
{{indent}}CUTLASS_CHECK(status);

{{indent}}return;
"""
)


def kernel_name(op, func_attrs):
    """Returns kernel_name given input cutlass op_instance and operator attrs."""

    from cutlass_lib import library

    threadblock = op.tile_description.procedural_name()
    extended_name = op.extended_name()
    opcode_class_name = library.OpcodeClassNames[
        op.tile_description.math_instruction.opcode_class
    ]
    layout = op.layout_name()
    align_ab = op.A.alignment
    align_c = op.C.alignment

    name = KERNEL_KEY_TEMPLATE.render(
        threadblock=threadblock,
        extended_name=extended_name,
        opcode_class_name=opcode_class_name,
        layout=layout,
        align_ab=align_ab,
        align_c=align_c,
    )
    return name.replace("\n", "")


def extract_config(f_proc_op, func_attrs):
    return common.extract_config(f_proc_op, partial(kernel_name, func_attrs=func_attrs))


def dual_gemm_instance(
    op_def: str, func_attrs: Dict[str, Any], for_profiler: bool
) -> str:
    tmp = op_def.replace(
        "GemmIdentityThreadblockSwizzle<8>", "GemmIdentityThreadblockSwizzle<1>"
    )
    return tmp


def emit_instance(
    op,
    for_profiler,
    f_instance_convertor=dual_gemm_instance,
    emit_kernel=False,
    func_attrs=None,
    broadcast_b1=False,
):
    import cutlass_lib

    emiter = cutlass_lib.gemm_operation.EmitDualGemmInstance()
    op_def = emiter.emit(op, broadcast_b1=broadcast_b1)
    op_def = f_instance_convertor(op_def, func_attrs, for_profiler)
    return op_def


def default_fproc(
    *,
    op,
    a_layout,
    b_layout,
    c_layout,
    epilogue_name,
    epilogue2_name,
    permute_layout=None,
    dtype="float16",
):
    import copy

    import cutlass_lib

    backend_spec = CUDASpec()
    data_type = backend_spec.dtype_to_lib_type(dtype)

    ret = []
    # skip simt kernels
    if (
        op.tile_description.math_instruction.opcode_class
        == cutlass_lib.library.OpcodeClass.Simt
    ):
        return ret

    if data_type == "float":
        if (
            op.tile_description.math_instruction.element_a
            != cutlass_lib.library.DataType.f32
            and op.tile_description.math_instruction.element_a
            != cutlass_lib.library.DataType.tf32
        ):
            return ret
    acc_type = cutlass_lib.library.DataType.f32
    # check target use fp16 acc
    if "use_fp16_acc" in Target.current()._kwargs and data_type == "cutlass::half_t":
        if Target.current()._kwargs["use_fp16_acc"]:
            acc_type = cutlass_lib.library.DataType.f16

    if (
        cutlass_lib.library.DataTypeTag[op.A.element] == data_type
        and cutlass_lib.library.DataTypeTag[op.B.element] == data_type
        and cutlass_lib.library.DataTypeTag[op.C.element] == data_type
        and op.accumulator_type() == acc_type
        and op.A.layout == a_layout
        and op.B.layout == b_layout
    ):
        op = copy.deepcopy(op)
        # set output major
        op.C.layout = c_layout
        # set epilogue
        op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[epilogue_name]
        op.epilogue_functor2 = cutlass_lib.library.EpilogueFunctorName[epilogue2_name]
        op.element_epilogue = acc_type
        if permute_layout is not None:
            op.permute_layout = cutlass_lib.library.EpiloguePermuteLayoutName[
                permute_layout
            ]
        # set C alignment
        alignments = alignment.get_alignments(dtype)
        for i in alignments:
            op = copy.deepcopy(op)
            op.C.alignment = i
            ret.append(op)
    return ret


def make_fproc(
    func_attrs,
    layout,
    dtype="float16",
):
    """
    This function sets a callback for processing the epilogue of the kernel
    associated with func_attrs.
    """

    def fproc(op):
        a_layout, b_layout, c_layout = layout.cutlass_lib_layouts()
        return default_fproc(
            op=op,
            a_layout=a_layout,
            b_layout=b_layout,
            c_layout=c_layout,
            epilogue_name=func_attrs["epilogue"],
            epilogue2_name=func_attrs["epilogue2"],
            dtype=dtype,
        )

    func_attrs["op_instance"] = extract_config(fproc, func_attrs)


def gen_function(
    func_attrs,
    src_template,
    exec_cond_template,
    problem_args,
    input_ndims,
    weight_ndims,
    output_ndims,
    dim_info_dict,
    f_instance_convertor=dual_gemm_instance,
    emit_kernel=False,
    support_split_k=False,
    input_addr_calculator="",
    output_addr_calculator="",
    extra_code="",
    broadcast_b1=False,
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]
    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = emit_instance(
                op_instance[algo],
                for_profiler=False,
                f_instance_convertor=f_instance_convertor,
                emit_kernel=emit_kernel,
                func_attrs=func_attrs,
                broadcast_b1=broadcast_b1,
            )
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = common.INSTANCE_TEMPLATE.render(
            config=config, name=fname, config_name=common.extract_config_name(config)
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst
    shape_eval_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = EXEC_TEMPLATE.render(
            indent="    ",
            instance=fname,
            problem_args=problem_args,
            support_split_k=support_split_k,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    input_output_checks = common.INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
    )

    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )

    return src_template.render(
        instances=instance_decl,
        function_name=func_name,
        dtype=elem_input_type,
        shape_eval=shape_eval_func,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        support_split_k=support_split_k,
        has_d=common.has_d(func_attrs),
        has_d1=common.has_d1(func_attrs),
        extra_code=extra_code,
    )


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    args_parser_template,
    emit_kernel=False,
    support_split_k=False,
    output_addr_calculator="",
    bias_ptr_arg=None,
    extra_code="",
    broadcast_b1=False,
    broadcasted_bdim_id=0,
    ndims=2,
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
    op_instance = func_attrs["op_instance"]

    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )

    has_bias = bias_ptr_arg is not None
    instance_name_base = "GemmInstance"
    exec_program = EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        support_split_k=support_split_k,
        problem_args=problem_args_template.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
            broadcast_b1=broadcast_b1,
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
        config = emit_instance(
            op,
            for_profiler=True,
            emit_kernel=emit_kernel,
            func_attrs=func_attrs,
            broadcast_b1=broadcast_b1,
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
            support_split_k=support_split_k,
            split_k="split_k",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
        )
        instances.append(instance)
        benchmark_instances.append(benchmark_instance)
    op_func = src_template.render(
        is_profiler=True,
        instances="\n".join(instances),
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_program,
        output_addr_calculator=output_addr_calculator,
        support_split_k=support_split_k,
        extra_code=extra_code,
    )
    benchmark_adims = ["a_dim" + str(i) for i in range(ndims)]
    benchmark_bdims = ["b_dim" + str(i) for i in range(ndims)]
    benchmark_cdims = ["c_dim" + str(i) for i in range(ndims)]
    func_call = common.FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name=function_name,
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b_ptr="memory_pool->RequestTensorByIdx(1)",
        has_bias=has_bias,
        bias_ptr=bias_ptr_arg,
        c_ptr="memory_pool->RequestTensorByIdx(2)",
        split_k="split_k",
        adims=benchmark_adims,
        bdims=benchmark_bdims,
        cdims=benchmark_cdims,
    )
    # TODO: Render args_parse by caller.
    args_parse = (
        args_parser_template
        if isinstance(args_parser_template, str)
        else args_parser_template.render()
    )
    code = common.PROFILER_TEMPLATE.render(
        op_func=op_func,
        has_bias=has_bias,
        support_split_k=support_split_k,
        args_parse=args_parse,
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        func_call=func_call,
        tensor_decl=TENSOR_DECL_TEMPLATE.render(
            has_bias=has_bias,
            adims=benchmark_adims,
            bdims=benchmark_bdims,
            cdims=benchmark_cdims,
            broadcast_b1=broadcast_b1,
            broadcasted_bdim_id=broadcasted_bdim_id,
        ),
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return common.build_profiler(file_pairs)
