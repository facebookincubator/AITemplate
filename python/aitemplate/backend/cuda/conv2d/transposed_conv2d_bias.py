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
transposed conv2d + bias + (relu) codegen
"""
import re

import jinja2

from aitemplate.backend.backend_spec import CUDASpec

from ... import registry
from . import common, common_conv2d_bias_activation as cba

# pylint: disable=C0103,C0415,W0613,C0301

SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

{{extra_header}}

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      auto msg = std::string("Got cutlass error: ") + cutlassGetStatusString(error) + \\
          " at: " + std::to_string(__LINE__);                                         \\
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
    void* bias_ptr,
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
    int stride,
    int dilation,
    int pad,
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
  TensorNHWC layout_B(TensorNHWC::packed(cutlass::make_Coord(i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch)));
  TensorNHWC layout_C(TensorNHWC::packed(cutlass::make_Coord(i32_out_batch, i32_out_h, i32_out_w, i32_out_ch)));

  cutlass::conv::Conv2dProblemSize problem_size(
        {i32_out_batch, i32_out_h, i32_out_w, i32_out_ch},
        {i32_out_ch, i32_kernel_h, i32_kernel_w, i32_in_ch},
        {pad, pad, pad, pad},
        {stride, stride},
        {dilation, dilation},
        {i32_batch, i32_in_h, i32_in_w, i32_in_ch},
        cutlass::conv::Mode::kCrossCorrelation,
        1
  );

  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this conv2d specialization."
  );
}
"""
)


def _conv_transpose_instance(op_def):
    tmp = op_def.replace("DefaultConv2dFprop", "DefaultConv2dDgrad")
    tmp = re.sub(
        r"cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<\d>",
        "cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>",
        tmp,
    )
    return tmp


def emit_instance(op, f_instance_convertor=_conv_transpose_instance):
    import cutlass_lib

    emiter = cutlass_lib.conv2d_operation.EmitConv2dInstance()
    op_def = emiter.emit(op)
    op_def = f_instance_convertor(op_def)
    return op_def


@registry.reg("cuda.transposed_conv2d_bias.config")
@registry.reg("cuda.transposed_conv2d_bias_relu.config")
def conv2d_config(func_attrs, dtype="float16"):
    func_attrs["op_instance"] = common.extract_config(func_attrs)


@registry.reg("cuda.transposed_conv2d_bias.gen_function")
@registry.reg("cuda.transposed_conv2d_bias_relu.gen_function")
def gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    return common.gen_function(
        func_attrs,
        cba.INSTANCE_TEMPLATE,
        cba.EXEC_TEMPLATE,
        SRC_TEMPLATE,
        exec_cond_remplate,
        shape_eval_template,
        shape_save_template,
        f_emit_instance=emit_instance,
    )


@registry.reg("cuda.transposed_conv2d_bias.func_decl")
@registry.reg("cuda.transposed_conv2d_bias_relu.func_decl")
def conv2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return cba.FUNC_DECL_TEMPLATE.render(func_name=func_name)


@registry.reg("cuda.transposed_conv2d_bias.func_call")
@registry.reg("cuda.transposed_conv2d_bias_relu.func_call")
def conv2d_gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    b = func_attrs["inputs"][2]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return cba.FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        out_ptr=y._attrs["name"],
        bias_ptr=b._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + wshape[0]._attrs["name"],
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_kernel_h="&" + wshape[1]._attrs["name"],
        p_kernel_w="&" + wshape[2]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        stride=func_attrs["stride"],
        dilation=func_attrs["dilate"],
        pad=func_attrs["pad"],
        indent=indent,
    )


@registry.reg("cuda.transposed_conv2d_bias.gen_profiler")
@registry.reg("cuda.transposed_conv2d_bias_relu.gen_profiler")
def gen_profiler(func_attrs, workdir, shape_template):
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    # shape func
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
        stride="stride",
        dilate="dilation",
        pad="pad",
    )
    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])
    file_pairs = []
    for op_name, op in op_instance.items():
        config = emit_instance(op)

        config_name = common.extract_config_name(config)
        name = "DeviceConvBwdInstance"
        instance = cba.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=name, config=config
        )
        exec_program = cba.EXEC_TEMPLATE.render(
            indent="  ", is_profiler=True, instance=name, dtype=dtype
        )
        op_func = SRC_TEMPLATE.render(
            instances=instance,
            function_name="conv",
            shape_func="",
            exec_paths=exec_program,
        )
        code = cba.PROFILER_TEMPLATE.render(
            op_func=op_func, shape_func=shape_func, name=name
        )
        common.add_profiler(file_pairs, workdir, op_type, op_name, code)
    # build
    return common.build_profiler(file_pairs)


@registry.reg("cuda.transposed_conv2d_bias.filter")
@registry.reg("cuda.transposed_conv2d_bias_relu.filter")
def conv2d_function_filter(cfg, func_attrs, x_shape):
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
