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
common template for conv2d
"""
import re

from collections import OrderedDict
from hashlib import sha1
from typing import List

import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal.common import add_profiler, build_profiler
from aitemplate.backend.target import Target

from aitemplate.utils import alignment


KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass{{opcode_class}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = cutlass::conv::device::ImplicitGemmConvolution<{{config_name}}>;
"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}using ElementComputeEpilogue = typename {{instance_name}}::ElementCompute;
{{indent}}//  TODO: cast to right dtype
{{indent}}typename {{instance_name}}::Arguments arguments{
{{indent}}    problem_size,                                                                 // ConvProblemSize const & problem_size
{{indent}}    {static_cast<{{dtype}}*>(in_ptr), layout_A},                                  // TensorRefA const & ref_A
{{indent}}    {static_cast<{{dtype}}*>(weight_ptr), layout_B},                              // TensorRefA const & ref_B
{% if is_bias %}
{{indent}}    {static_cast<{{dtype}}*>(bias_ptr), cutlass::layout::TensorNHWC::Stride(0)},  // TensorRefC const & ref_C
{% elif is_bias_add %}
{{indent}}    {static_cast<{{dtype}}*>(res_ptr), layout_C},                                 // TensorRefC const & ref_C
{% else %}
{{indent}}    {static_cast<{{dtype}}*>(out_ptr), layout_C},                                 // TensorRefC const & ref_C
{% endif %}
{{indent}}    {static_cast<{{dtype}}*>(out_ptr), layout_C},                                 // TensorRefC const & ref_D
{% if is_bias %}
{{indent}}    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},                       // typename EpilogueOutputOp::Params const & output_op
{% elif is_bias_add %}
{{indent}}    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},                       // typename EpilogueOutputOp::Params const & output_op
{{indent}}    cutlass::conv::SplitKMode::kSerial,                                           // SplitKMode const & split_k_mode
{{indent}}    static_cast<{{dtype}}*>(bias_ptr),                                            // void * ptr_Vector
{{indent}}    nullptr,                                                                      // void * ptr_Tensor
{{indent}}    0,                                                                            // typename LayoutC::Stride::Index ldr
{{indent}}    *out_ch,                                                                      // typename LayoutC::Stride::Index ldt
{% else %}
{{indent}}    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},                       // typename EpilogueOutputOp::Params const & output_op
{% endif %}
{{indent}}};
{{indent}}{{instance_name}} conv_op;
{% if is_profiler %}
{{indent}}size_t workspace_size = conv_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE_{{instance_name}} = workspace_size;
{% endif %}
{{indent}}auto status = conv_op.can_implement(arguments);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = conv_op.initialize(arguments, workspace);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = conv_op(stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <cstdio>
#include <stdexcept>

#include "cutlass/cutlass.h"
{% if is_transpose %}
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
{% elif is_depthwise %}
#include "cutlass/conv/kernel/default_depthwise_fprop.h"
{% else %}
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_group_fprop.h"
{% endif %}
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

{{extra_header}}

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      static char msg[2048];                                                          \\
      snprintf(msg, sizeof(msg), "[%s] Got cutlass error: %s at: %s",                 \\
        __FILE__, cutlassGetStatusString(error), __LINE__);                           \\
      fprintf(stderr, msg);                                                           \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}

{{functions}}
"""
)

FUNCTION_TEMPLATE = jinja2.Template(
    """
void {{function_name}} (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,
{% if is_bias %}
    void* bias_ptr,
{% elif is_bias_add %}
    void* bias_ptr,
    void* res_ptr,
{% endif %}
    uint8_t* workspace,
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int strideh,
    int dilationh,
    int padh,
    int stridew,
    int dilationw,
    int padw,
    cudaStream_t stream
  ) {

  {{shape_function}}

  int i32_batch = *batch;
  int i32_in_h = *in_h;
  int i32_in_w = *in_w;
  int i32_in_ch = *in_ch;
  int i32_out_ch = *out_ch;
  int i32_kernel_h = *kernel_h;
  int i32_kernel_w = *kernel_w;
  int i32_out_batch = *out_batch;
  int i32_out_h = *out_h;
  int i32_out_w = *out_w;

  using cutlass::layout::TensorNHWC;
  TensorNHWC layout_A(TensorNHWC::packed(cutlass::make_Coord(i32_batch, i32_in_h, i32_in_w, i32_in_ch)));
{% if is_depthwise%}
  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_out_ch, i32_kernel_h, i32_kernel_w, 1)));
{% elif is_transpose %}
  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_in_ch, i32_kernel_h, i32_kernel_w, i32_out_ch)));
{% else %}
  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch)));
{% endif %}
  TensorNHWC layout_C(TensorNHWC::packed(cutlass::make_Coord(i32_out_batch, i32_out_h, i32_out_w, i32_out_ch)));

  cutlass::conv::Conv2dProblemSize problem_size(
{% if is_transpose %}
    {i32_out_batch, i32_out_h, i32_out_w, i32_out_ch},    // cutlass::Tensor4DCoord input_size
{% else %}
    {i32_batch, i32_in_h, i32_in_w, i32_in_ch},           // cutlass::Tensor4DCoord input_size
{% endif %}
{% if is_depthwise%}
    {i32_out_ch, i32_kernel_h, i32_kernel_w, 1},  // cutlass::Tensor4DCoord filter_size
{% elif is_transpose%}
    {i32_in_ch, i32_kernel_h, i32_kernel_w, i32_out_ch},  // cutlass::Tensor4DCoord filter_size
{% else %}
    {i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch},  // cutlass::Tensor4DCoord filter_size
{% endif %}
    {padh, padh, padw, padw},                                 // cutlass::Tensor4DCoord padding
    {strideh, stridew},                                     // cutlass::MatrixCoord stride
    {dilationh, dilationw},                                 // cutlass::MatrixCoord dilation
{% if is_transpose %}
    {i32_batch, i32_in_h, i32_in_w, i32_in_ch},           // cutlass::Tensor4DCoord output_size
{% else %}
    {i32_out_batch, i32_out_h, i32_out_w, i32_out_ch},    // cutlass::Tensor4DCoord output_size
{% endif %}
    cutlass::conv::Mode::kCrossCorrelation,               // cutlass::conv::Mode mode
    1                                                     // int split_k_slices
  );

  {{exec_paths}}

  throw std::runtime_error(
    "Unsupported workload for this conv2d specialization."
  );
}
"""
)

BENCHMARK_INSTANCE_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}  int ret = 0;
{{indent}}  try {
{{indent}}    ret = {{func_name}}(
{{indent}}      &runtime,
{{indent}}      &workspace_size,
{{indent}}      {{ni}},
{{indent}}      {{hi}},
{{indent}}      {{wi}},
{{indent}}      {{ci}},
{{indent}}      {{co}},
{{indent}}      {{kh}},
{{indent}}      {{kw}},
{{indent}}      {{no}},
{{indent}}      {{ho}},
{{indent}}      {{wo}},
{{indent}}      {{strideh}},
{{indent}}      {{dilationh}},
{{indent}}      {{padh}},
{{indent}}      {{stridew}},
{{indent}}      {{dilationw}},
{{indent}}      {{padw}},
{{indent}}      global_workspace_,
{{indent}}      stream
{{indent}}    );
{{indent}}  } catch (...) {
{{indent}}    runtime = 0;
{{indent}}    workspace_size = 0;
{{indent}}  }
{{indent}}  if (ret != 0)
{{indent}}    return ret;
{{indent}}  std::cout << "OP:{{conv_op_name}},"
{{indent}}            << "TIME:" << runtime << ","
{{indent}}            << "WS:" << workspace_size << std::endl;
{{indent}}}
"""
)

BENCHMARK_DECL_TEMPLATE = jinja2.Template(
    """
int benchmark_{{function_name}} (
  float*,
  size_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  int,
  int,
  int,
  int,
  int,
  int,
  uint8_t*,
  cudaStream_t
);
"""
)

BENCHMARK_TEMPLATE = jinja2.Template(
    """
int benchmark_{{function_name}} (
  float* runtime,
  size_t* workspace_size,
  int64_t NI,
  int64_t HI,
  int64_t WI,
  int64_t CI,
  int64_t CO,
  int64_t KH,
  int64_t KW,
  int64_t NO,
  int64_t HO,
  int64_t WO,
  int strideh,
  int dilationh,
  int padh,
  int stridew,
  int dilationw,
  int padw,
  uint8_t* global_workspace_,
  cudaStream_t stream
) {
  using ElementInputA = typename {{instance_name}}::ElementA;
  using ElementInputB = typename {{instance_name}}::ElementB;
  using ElementOutput = typename {{instance_name}}::ElementC;

  cutlass::HostTensor<ElementInputA, typename {{instance_name}}::LayoutA> x({NI, HI, WI, CI});
  cutlass::HostTensor<ElementInputB, typename {{instance_name}}::LayoutB> w({CO, KH, KW, CI});
{% if is_bias %}
  cutlass::HostTensor<ElementInputB, typename {{instance_name}}::LayoutB> b({(int)CO, 1, 1, 1});
{% elif is_bias_add %}
  cutlass::HostTensor<ElementInputB, typename {{instance_name}}::LayoutB> b({(int)CO, 1, 1, 1});
  cutlass::HostTensor<ElementOutput, typename {{instance_name}}::LayoutC> r({NO, HO, WO, CO});
{% endif %}
  cutlass::HostTensor<ElementOutput, typename {{instance_name}}::LayoutC> y({NO, HO, WO, CO});

  // warmup
{{func_call}}
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 5; ++i) {
{{func_call}}
  }
  cudaEventRecord(events[1], stream);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  *runtime = runtime_ms;
  *workspace_size = GLOBAL_WORKSPACE_SIZE_{{instance_name}};
  return 0;
}
"""
)

PROFILER_BENCHMARK_TEMPLATE = jinja2.Template(
    """
static size_t GLOBAL_WORKSPACE_SIZE_{{instance_name}} = 0;

{{op_source}}

{{benchmark}}
"""
)

PROFILER_MAIN_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"

{{benchmark_decls}}

int main(int argc, char** argv) {
  int64_t batch = std::stoi(argv[1]);
  int64_t in_h = std::stoi(argv[2]);
  int64_t in_w = std::stoi(argv[3]);
  int64_t in_ch = std::stoi(argv[4]);
  int64_t kernel_h = std::stoi(argv[5]);
  int64_t kernel_w = std::stoi(argv[6]);
  int64_t out_ch = std::stoi(argv[7]);
  int strideh = std::stoi(argv[8]);
  int padh = std::stoi(argv[9]);
  int dilationh = std::stoi(argv[10]);
  int stridew = std::stoi(argv[11]);
  int padw = std::stoi(argv[12]);
  int dilationw = std::stoi(argv[13]);

{{shape_func}}

  float runtime = 0;
  size_t workspace_size = 0;
  uint8_t* global_workspace_ = nullptr;
  cudaStream_t stream = nullptr;

{{benchmark_instances}}

  return 0;
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
{% if is_bias %}
  void*,
{% elif is_bias_add %}
  void*,
  void*,
{% endif %}
  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  cudaStream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{weight_ptr}},
{{indent}}    {{out_ptr}},
{% if is_bias %}
{{indent}}    {{bias_ptr}},
{% elif is_bias_add %}
{{indent}}    {{bias_ptr}},
{{indent}}    {{res_ptr}},
{% endif %}
{{indent}}    global_workspace_,
{{indent}}    {{p_batch}},
{{indent}}    {{p_out_ch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_kernel_h}},
{{indent}}    {{p_kernel_w}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{strideh}},
{{indent}}    {{dilationh}},
{{indent}}    {{padh}},
{{indent}}    {{stridew}},
{{indent}}    {{dilationw}},
{{indent}}    {{padw}},
{{indent}}    stream
{{indent}});
"""
)


def kernel_name(op, layout=None):
    """generate cuda kernel name"""
    from cutlass_lib import library

    threadblock = op.tile_description.procedural_name()
    extended_name = op.extended_name()
    opcode_class_name = library.OpcodeClassNames[
        op.tile_description.math_instruction.opcode_class
    ]
    if layout is None:
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


def emit_instance(op):
    """emit instance"""
    import cutlass_lib

    if hasattr(op, "binary_op"):
        emiter = cutlass_lib.conv2d_operation.EmitConv2dWithBroadcastInstance()
    else:
        emiter = cutlass_lib.conv2d_operation.EmitConv2dInstance()
    op_def = emiter.emit(op)
    return op_def


def extract_config(
    func_attrs,
    dtype="float16",
    skip_simt_kernels=False,
    f_apply_special_config=None,
    op_kind=None,
    op_layout=None,
):
    """Extracts cutlass config for conv kernels."""
    import copy

    import cutlass_lib

    spec = CUDASpec()
    lib_dtype = spec.dtype_to_lib_type(dtype)

    if lib_dtype == "float":
        data_type = cutlass_lib.library.DataType.f32
        acc_type = cutlass_lib.library.DataType.f32
    elif "half" in lib_dtype:
        data_type = cutlass_lib.library.DataType.f16
        acc_type = cutlass_lib.library.DataType.f32
        # check target use fp16 acc
        if "use_fp16_acc" in Target.current()._kwargs:
            if Target.current()._kwargs["use_fp16_acc"]:
                acc_type = cutlass_lib.library.DataType.f16
    elif "bfloat16" in lib_dtype:
        data_type = cutlass_lib.library.DataType.bf16
        acc_type = cutlass_lib.library.DataType.f32
        # check target use fp16 acc
        if "use_fp16_acc" in Target.current()._kwargs:
            if Target.current()._kwargs["use_fp16_acc"]:
                acc_type = cutlass_lib.library.DataType.bf16
    else:
        raise RuntimeError(f"Unsupported dtype {lib_dtype}")

    def f_proc_op(op):
        ret = []
        if (
            skip_simt_kernels
            and op.tile_description.math_instruction.opcode_class
            == cutlass_lib.library.OpcodeClass.Simt
        ):
            return ret

        if (
            op.A.element == data_type
            and op.B.element == data_type
            and op.C.element == data_type
            and op.iterator_algorithm == cutlass_lib.library.IteratorAlgorithm.Optimized
            and op.tile_description.math_instruction.element_accumulator == acc_type
        ):
            op = copy.deepcopy(op)

            # set epilogue
            epilogue_name = func_attrs["epilogue"]
            op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[epilogue_name]
            op.element_epilogue = acc_type

            # apply special config if required
            if f_apply_special_config is not None:
                op = f_apply_special_config(func_attrs, op)

            # set C alignment depending on the dtype
            for i in alignment.get_alignments(dtype):
                op = copy.deepcopy(op)
                op.C.alignment = i
                ret.append(op)

        return ret

    if op_kind is None:
        op_kind = cutlass_lib.library.OperationKind.Conv2d
    extract_ops = list(Target.current()._operators[op_kind].items())
    conv_kind = cutlass_lib.library.ConvKind.Fprop

    conv_ops = OrderedDict()
    for _, value in extract_ops:
        op = value[0]
        if op.conv_kind == conv_kind:
            ret = f_proc_op(op)
            if len(ret) > 0:
                for op_inst in ret:
                    key = kernel_name(op_inst, layout=op_layout)
                    conv_ops[key] = op_inst
    return conv_ops


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
    f_emit_instance=emit_instance,
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
    instance_name_base="DeviceConvFwdInstance",
):
    """Generate profiler sources."""
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]

    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])

    func_call_extra_args = {}
    if is_bias:
        func_call_extra_args = {
            "bias_ptr": "b.device_data()",
        }
    elif is_bias_add:
        func_call_extra_args = {
            "bias_ptr": "b.device_data()",
            "res_ptr": "r.device_data()",
        }

    benchmark_decls = []
    benchmark_instances = []
    profiler_benchmarks = {}

    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = f_emit_instance(op)
        config_name = extract_config_name(config)
        instance_name = f"{instance_name_base}_{instance_idx}"
        function_name = f"{op_type}_{op_name}"

        exec_program = EXEC_TEMPLATE.render(
            indent="  ",
            is_profiler=True,
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            instance_name=instance_name,
            dtype=dtype,
        )
        instance = INSTANCE_TEMPLATE.render(
            config_name=config_name,
            name=instance_name,
            config=config,
        )
        function = FUNCTION_TEMPLATE.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            is_transpose=is_transpose,
            is_depthwise=is_depthwise,
            function_name=function_name,
            shape_function="",
            exec_paths=exec_program,
        )
        op_source = SRC_TEMPLATE.render(
            is_transpose=is_transpose,
            is_depthwise=is_depthwise,
            extra_header=extra_header,
            instances=instance,
            functions=function,
        )

        func_call = FUNC_CALL_TEMPLATE.render(
            indent="  ",
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            func_name=function_name,
            in_ptr="x.device_data()",
            weight_ptr="w.device_data()",
            out_ptr="y.device_data()",
            **func_call_extra_args,
            p_batch="&NI",
            p_out_ch="&CO",
            p_in_ch="&CI",
            p_kernel_h="&KH",
            p_kernel_w="&KW",
            p_in_h="&HI",
            p_in_w="&WI",
            p_out_batch="&NO",
            p_out_h="&HO",
            p_out_w="&WO",
            strideh="strideh",
            dilationh="dilationh",
            padh="padh",
            stridew="stridew",
            dilationw="dilationw",
            padw="padw",
        )
        benchmark = BENCHMARK_TEMPLATE.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            instance_name_base=instance_name_base,
            function_name=function_name,
            func_call=func_call,
            instance_name=instance_name,
        )

        profiler_benchmarks[function_name] = PROFILER_BENCHMARK_TEMPLATE.render(
            op_source=op_source,
            benchmark=benchmark,
            instance_name=instance_name,
        )

        benchmark_instance = BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            conv_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            ni="NI",
            hi="HI",
            wi="WI",
            ci="CI",
            co="CO",
            kh="KH",
            kw="KW",
            no="NO",
            ho="HO",
            wo="WO",
            strideh="SH",
            dilationh="DH",
            padh="PH",
            stridew="SW",
            dilationw="DW",
            padw="PW",
        )
        benchmark_instances.append(benchmark_instance)

        benchmark_decl = BENCHMARK_DECL_TEMPLATE.render(
            function_name=function_name,
        )
        benchmark_decls.append(benchmark_decl)

    shape_func = shape_template.render(
        indent="  ",
        dtype="int64_t ",
        div="/",
        x_dim0="batch",
        x_dim1="in_h",
        x_dim2="in_w",
        x_dim3="in_ch",
        w_dim0="out_ch",
        w_dim1="kernel_h",
        w_dim2="kernel_w",
        strideh="strideh",
        dilateh="dilationh",
        padh="padh",
        stridew="stridew",
        dilatew="dilationw",
        padw="padw",
    )
    profiler_main_code = PROFILER_MAIN_TEMPLATE.render(
        shape_func=shape_func,
        benchmark_decls="\n".join(benchmark_decls),
        benchmark_instances="\n".join(benchmark_instances),
    )

    code = {profiler_filename: profiler_main_code}
    for benchmark_filename, benchmark_code in profiler_benchmarks.items():
        code[benchmark_filename] = benchmark_code

    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    add_profiler(file_pairs, workdir, op_type, profiler_filename, code)

    # build
    return build_profiler(file_pairs)


def extract_config_name(config):
    """Extracts config name from a given config."""
    pattern = re.compile(r"\s*using\s(.*?)\s=")
    decl = config.split("\n")[2]
    match = pattern.match(decl)
    if match is None:
        raise RuntimeError("Invalid config: \n" + config)
    return match.groups()[0]


def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
    f_emit_instance=emit_instance,
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
):
    """Function definition codegen."""
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        emitted_instance = f_emit_instance(op_instance[value])
        if value not in inst_def_flag:
            inst_def_flag.add(value)
            config = emitted_instance
        else:
            config = ""
        inst = INSTANCE_TEMPLATE.render(
            config=config,
            name=fname,
            config_name=extract_config_name(emitted_instance),
        )
        instances[key] = inst
        instance_decl += inst

    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        w_dim0="*out_ch",
        w_dim1="*kernel_h",
        w_dim2="*kernel_w",
        strideh="strideh",
        dilateh="dilationh",
        padh="padh",
        stridew="stridew",
        dilatew="dilationw",
        padw="padw",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
        y_dim3="*out_ch",
    )
    shape_func = shape_eval_func + shape_save_func

    exec_paths = ""
    for key in instances:
        fname = "f" + sha1(key.encode()).hexdigest()
        program = EXEC_TEMPLATE.render(
            is_bias=is_bias,
            is_bias_add=is_bias_add,
            indent=" " * 4,
            instance_name=fname,
            dtype=dtype,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst

    function = FUNCTION_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        is_transpose=is_transpose,
        is_depthwise=is_depthwise,
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
    )

    return SRC_TEMPLATE.render(
        is_transpose=is_transpose,
        is_depthwise=is_depthwise,
        extra_header=extra_header,
        instances=instance_decl,
        functions=function,
    )


def gen_function_decl(
    func_attrs,
    is_bias=False,
    is_bias_add=False,
):
    func_name = func_attrs["name"]

    return FUNC_DECL_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        func_name=func_name,
    )


def gen_function_call(
    func_attrs,
    indent="  ",
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]

    func_call_extra_args = {}
    if is_bias:
        b = func_attrs["inputs"][2]
        func_call_extra_args = {
            "bias_ptr": b._attrs["name"],
        }
    elif is_bias_add:
        b = func_attrs["inputs"][2]
        r = func_attrs["inputs"][3]
        func_call_extra_args = {
            "bias_ptr": b._attrs["name"],
            "res_ptr": r._attrs["name"],
        }

    out_ch = wshape[-1]._attrs["name"] if is_transpose else wshape[0]._attrs["name"]
    return FUNC_CALL_TEMPLATE.render(
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        out_ptr=y._attrs["name"],
        **func_call_extra_args,
        p_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + out_ch,
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_kernel_h="&" + wshape[1]._attrs["name"],
        p_kernel_w="&" + wshape[2]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        strideh=func_attrs["stride"]
        if isinstance(func_attrs["stride"], int)
        else func_attrs["stride"][0],
        dilationh=func_attrs["dilate"]
        if isinstance(func_attrs["dilate"], int)
        else func_attrs["dilate"][0],
        padh=func_attrs["pad"]
        if isinstance(func_attrs["pad"], int)
        else func_attrs["pad"][0],
        stridew=func_attrs["stride"]
        if isinstance(func_attrs["stride"], int)
        else func_attrs["stride"][1],
        dilationw=func_attrs["dilate"]
        if isinstance(func_attrs["dilate"], int)
        else func_attrs["dilate"][1],
        padw=func_attrs["pad"]
        if isinstance(func_attrs["pad"], int)
        else func_attrs["pad"][1],
        indent=indent,
    )


def _cal_align_ab(x_shape: List[int], dtype="float16") -> int:
    """Returns input alignment."""
    k = x_shape[3]  # CI
    return alignment.find_max_alignment(k, dtype)


def function_filter(
    cfg,
    func_attrs,
    x_shape,
):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    dtype = func_attrs["inputs"][0]._attrs["dtype"]
    ab_alignment = _cal_align_ab(x_shape, dtype=dtype)

    tmp = cfg.split("_")
    align_c = int(tmp[-1])
    align_ab = int(tmp[-2])

    if align_c != func_attrs["epilogue_alignment"]:
        return False
    if align_ab != ab_alignment:
        return False

    return True
