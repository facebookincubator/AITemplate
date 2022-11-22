#  Copyright (c) Meta Platform, Inc. and its affiliates.
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
Codegen for conv3d.
"""
import jinja2

from aitemplate.backend.backend_spec import CUDASpec

from ... import registry
from . import common

# pylint: disable=C0103,C0415,W0613,C0301

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = cutlass::conv::device::ImplicitGemmConvolution<{{config_name}}>;
"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}using ElementComputeEpilogue = typename {{instance}}::ElementCompute;
//  TODO: cast to right dtype
{{indent}}typename {{instance}}::Arguments arguments{
{{indent}}    problem_size,
{{indent}}    {static_cast<{{dtype}}*>(in_ptr), layout_A},
{{indent}}    {static_cast<{{dtype}}*>(weight_ptr), layout_B},
{{indent}}    {static_cast<{{dtype}}*>(out_ptr), layout_C},
{{indent}}    {static_cast<{{dtype}}*>(out_ptr), layout_C},
{{indent}}    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
{{indent}}};
{{indent}}{{instance}} implicit_gemm_op;
{% if is_profiler %}
{{indent}}size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% endif %}
{{indent}}auto status = implicit_gemm_op.can_implement(arguments);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = implicit_gemm_op.initialize(arguments, workspace);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = implicit_gemm_op(stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <string>
#include <stdexcept>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

{{extra_header}}

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

{{instances_def}}

void {{function_name}} (
    void* in_ptr,
    void* weight_ptr,
    void* out_ptr,
    uint8_t* workspace,
    int64_t* batch,
    int64_t* out_ch,
    int64_t* in_ch,
    int64_t* kernel_d,
    int64_t* kernel_h,
    int64_t* kernel_w,
    int64_t* in_d,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_d,
    int64_t* out_h,
    int64_t* out_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int pad_d,
    int pad_h,
    int pad_w,
    cudaStream_t stream
  ) {

  {{shape_function}}
  int i32_batch = *batch;
  int i32_in_d = *in_d;
  int i32_in_h = *in_h;
  int i32_in_w = *in_w;
  int i32_in_ch = *in_ch;
  int i32_out_ch = *out_ch;
  int i32_kernel_d = *kernel_d;
  int i32_kernel_h = *kernel_h;
  int i32_kernel_w = *kernel_w;
  int i32_out_batch = *out_batch;
  int i32_out_d = *out_d;
  int i32_out_h = *out_h;
  int i32_out_w = *out_w;

  using cutlass::layout::TensorNDHWC;
  TensorNDHWC layout_A(TensorNDHWC::packed(cutlass::make_Coord(i32_batch, i32_in_d, i32_in_h, i32_in_w, i32_in_ch)));
  TensorNDHWC layout_B(TensorNDHWC::packed(cutlass::make_Coord(i32_out_ch, i32_kernel_d, i32_kernel_h, i32_kernel_w, i32_in_ch)));
  TensorNDHWC layout_C(TensorNDHWC::packed(cutlass::make_Coord(i32_out_batch, i32_out_d, i32_out_h, i32_out_w, i32_out_ch)));

  cutlass::conv::Conv3dProblemSize problem_size(
    cutlass::Tensor5DCoord(i32_batch, i32_in_d, i32_in_h, i32_in_w, i32_in_ch),
    cutlass::Tensor5DCoord(i32_out_ch, i32_kernel_d, i32_kernel_h, i32_kernel_w, i32_in_ch),
    cutlass::make_Coord(pad_d, pad_h, pad_w),
    cutlass::make_Coord(stride_d, stride_h, stride_w),
    cutlass::make_Coord(dilation_d, dilation_h, dilation_w),
    cutlass::conv::Mode::kCrossCorrelation,
    1,
    1
  );

  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this conv3d specialization."
  );
}
"""
)


PROFILER_TEMPLATE = jinja2.Template(
    """
size_t GLOBAL_WORKSPACE_SIZE = 0;

{{op_func}}

int main(int argc, char** argv) {
  int64_t batch = std::stoi(argv[1]);
  int64_t in_d = std::stoi(argv[2]);
  int64_t in_h = std::stoi(argv[3]);
  int64_t in_w = std::stoi(argv[4]);
  int64_t in_ch = std::stoi(argv[5]);
  int64_t kernel_d = std::stoi(argv[6]);
  int64_t kernel_h = std::stoi(argv[7]);
  int64_t kernel_w = std::stoi(argv[8]);
  int64_t out_ch = std::stoi(argv[9]);
  int stride_d = std::stoi(argv[10]);
  int stride_h = std::stoi(argv[11]);
  int stride_w = std::stoi(argv[12]);
  int pad_d = std::stoi(argv[13]);
  int pad_h = std::stoi(argv[14]);
  int pad_w = std::stoi(argv[15]);
  int dilation_d = std::stoi(argv[16]);
  int dilation_h = std::stoi(argv[17]);
  int dilation_w = std::stoi(argv[18]);
  {{shape_func}}
  using ElementOutput = typename {{name}}::ElementC;
  using ElementInputA = typename {{name}}::ElementA;
  using ElementInputB = typename {{name}}::ElementB;

  uint8_t* global_workspace = nullptr;
  cudaStream_t stream = nullptr;

  cutlass::HostTensor<ElementInputA, typename {{name}}::LayoutA> x({NI, DI, HI, WI, CI});
  cutlass::HostTensor<ElementInputB, typename {{name}}::LayoutB> w({CO, KD, KH, KW, CI});
  cutlass::HostTensor<ElementOutput, typename {{name}}::LayoutC> y({NO, DO, HO, WO, CO});

  //
  // warmup
  conv(x.device_data(),
       w.device_data(),
       y.device_data(),
       global_workspace,
       &NI,
       &CO,
       &CI,
       &KD,
       &KH,
       &KW,
       &DI,
       &HI,
       &WI,
       &NO,
       &DO,
       &HO,
       &WO,
       stride_d,
       stride_h,
       stride_w,
       dilation_d,
       dilation_h,
       dilation_w,
       pad_d,
       pad_h,
       pad_w,
       stream);
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0]);
  for (int i = 0; i < 5; ++i) {
      conv(x.device_data(),
       w.device_data(),
       y.device_data(),
       global_workspace,
       &NI,
       &CO,
       &CI,
       &KD,
       &KH,
       &KW,
       &DI,
       &HI,
       &WI,
       &NO,
       &DO,
       &HO,
       &WO,
       stride_d,
       stride_h,
       stride_w,
       dilation_d,
       dilation_h,
       dilation_w,
       pad_d,
       pad_h,
       pad_w,
       stream);
  }
  cudaEventRecord(events[1]);
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
  std::cout << "TIME:" << runtime_ms << std::endl;
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
}

"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
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
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
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
{{indent}}    global_workspace_,
{{indent}}    {{p_batch}},
{{indent}}    {{p_out_ch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_kernel_d}},
{{indent}}    {{p_kernel_h}},
{{indent}}    {{p_kernel_w}},
{{indent}}    {{p_in_d}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_d}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{stride_d}},
{{indent}}    {{stride_h}},
{{indent}}    {{stride_w}},
{{indent}}    {{dilation_d}},
{{indent}}    {{dilation_h}},
{{indent}}    {{dilation_w}},
{{indent}}    {{pad_d}},
{{indent}}    {{pad_h}},
{{indent}}    {{pad_w}},
{{indent}}    stream
{{indent}});
"""
)


@registry.reg("cuda.conv3d.config")
def conv3d_config(func_attrs, dtype="float16"):
    """Populates conv3d cutlass configs into 'op_instance' field."""
    func_attrs["op_instance"] = common.extract_config(func_attrs)


@registry.reg("cuda.conv3d.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    """Codegen for conv3d profiler."""
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    # shape func
    shape_func = shape_template.render(
        indent="  ",
        dtype="int64_t ",
        div="/",
        x_dim0="batch",
        x_dim1="in_d",
        x_dim2="in_h",
        x_dim3="in_w",
        x_dim4="in_ch",
        w_dim0="out_ch",
        w_dim1="kernel_d",
        w_dim2="kernel_h",
        w_dim3="kernel_w",
        stride_d="stride_d",
        stride_h="stride_h",
        stride_w="stride_w",
        dilate_d="dilation_d",
        dilate_h="dilation_h",
        dilate_w="dilation_w",
        pad_d="pad_d",
        pad_h="pad_h",
        pad_w="pad_w",
    )
    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])
    file_pairs = []
    for op_name, op in op_instance.items():
        config = common.emit_instance(op)
        config_name = common.extract_config_name(config)
        name = "DeviceConvFwdInstance"
        instance = INSTANCE_TEMPLATE.render(
            config_name=config_name, name=name, config=config
        )
        exec_program = EXEC_TEMPLATE.render(
            indent="  ", is_profiler=True, instance=name, dtype=dtype
        )
        op_func = SRC_TEMPLATE.render(
            instances=instance,
            function_name="conv",
            shape_func="",
            exec_paths=exec_program,
        )
        code = PROFILER_TEMPLATE.render(
            op_func=op_func, shape_func=shape_func, name=name
        )
        common.add_profiler(file_pairs, workdir, op_type, op_name, code)
    # build
    return common.build_profiler(file_pairs)


@registry.reg("cuda.conv3d.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    """Codegen for conv3d function."""
    return common.gen_function(
        func_attrs,
        INSTANCE_TEMPLATE,
        EXEC_TEMPLATE,
        SRC_TEMPLATE,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
    )


@registry.reg("cuda.conv3d.func_decl")
def conv3d_gen_function_decl(func_attrs):
    """Codegen for conv3d function declaration."""
    func_name = func_attrs["name"]
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.conv3d.func_call")
def conv3d_gen_function_call(func_attrs, indent="  "):
    """Codegen for conv3d function call."""
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + wshape[0]._attrs["name"],
        p_in_ch="&" + xshape[4]._attrs["name"],
        p_kernel_d="&" + wshape[1]._attrs["name"],
        p_kernel_h="&" + wshape[2]._attrs["name"],
        p_kernel_w="&" + wshape[3]._attrs["name"],
        p_in_d="&" + xshape[1]._attrs["name"],
        p_in_h="&" + xshape[2]._attrs["name"],
        p_in_w="&" + xshape[3]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_d="&" + yshape[1]._attrs["name"],
        p_out_h="&" + yshape[2]._attrs["name"],
        p_out_w="&" + yshape[3]._attrs["name"],
        stride_d=func_attrs["stride"][0],
        stride_h=func_attrs["stride"][1],
        stride_w=func_attrs["stride"][2],
        dilation_d=func_attrs["dilate"][0],
        dilation_h=func_attrs["dilate"][1],
        dilation_w=func_attrs["dilate"][2],
        pad_d=func_attrs["pad"][0],
        pad_h=func_attrs["pad"][1],
        pad_w=func_attrs["pad"][2],
        indent=indent,
    )


@registry.reg("cuda.conv3d.filter")
def conv3d_function_filter(cfg, func_attrs, x_shape):
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
    return common.function_filter(cfg, func_attrs, x_shape)
