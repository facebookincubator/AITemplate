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
ROCM codegen functions for pool2d.
"""
from hashlib import sha1

import jinja2

# pylint: disable=C0103,C0301,W0613,W0612

INSTANCE_TEMPLATE = jinja2.Template(
    """
using {{name}} = ck::tensor_operation::device::DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C<
ck::half_t, ck::half_t, float, {{reduce_func}}, false, 64, 64, 1, 4, 1, 4>;
"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}auto op =  {{instance}}{};
{{indent}}auto invoker_ptr  = op.MakeInvokerPointer();
{{indent}}auto argument_ptr = op.MakeArgumentPointer(static_cast<ck::half_t *>(in_ptr),
{{indent}}                                           static_cast<ck::half_t *>(out_ptr),
{{indent}}                                           nullptr,
{{indent}}                                           *batch,
{{indent}}                                           *in_ch,
{{indent}}                                           input_shape,
{{indent}}                                           kernel_shape,
{{indent}}                                           output_shape,
{{indent}}                                           conv_filter_strides,
{{indent}}                                           input_left_pads,
{{indent}}                                           input_right_pads);
{{indent}}if(!op.IsSupportedArgument(argument_ptr.get())) {
{{indent}}  LOG(FATAL) << "wrong! " << op.GetTypeString() << " with the specified compilation parameters does not support this Pool problem.";
{{indent}}}
{{indent}}invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "logging.h"
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/reduction_operator.hpp"
#include "include/ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_nhwc_nhwc.hpp"

{{instances}}


void {{function_name}}(
    void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_ch,
    int64_t* in_h,
    int64_t* in_w,
    int64_t* out_batch,
    int64_t* out_h,
    int64_t* out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t pad,
    hipStream_t stream
    ) {
  {{shape_function}}

  const std::array<ck::index_t, 2> conv_filter_strides{static_cast<ck::index_t>(stride),
    static_cast<ck::index_t>(stride)};
  const std::array<ck::index_t, 2> input_left_pads{static_cast<ck::index_t>(pad),
    static_cast<ck::index_t>(pad)};
  const std::array<ck::index_t, 2> input_right_pads{static_cast<ck::index_t>(pad),
    static_cast<ck::index_t>(pad)};
  const std::array<ck::index_t, 2> input_shape{static_cast<ck::index_t>(*in_h),
    static_cast<ck::index_t>(*in_w)};
  const std::array<ck::index_t, 2> kernel_shape{static_cast<ck::index_t>(kernel_h),
    static_cast<ck::index_t>(kernel_w)};
  const std::array<ck::index_t, 2> output_shape{static_cast<ck::index_t>(*out_h),
    static_cast<ck::index_t>(*out_w)};

  {{exec_paths}}

  throw std::runtime_error(
      "Unsupported workload for this conv2d specialization."
  );
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  hipStream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{kernel_h}},
{{indent}}    {{kernel_w}},
{{indent}}    {{stride}},
{{indent}}    {{pad}},
{{indent}}    stream
{{indent}});
"""
)


def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    """
    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    exec_cond_template : jinja2.Template
        Generates if statement to execute kernel.
    shape_eval_template : jinja2.Template
        Generates shape calculation.
        The template is passed from compiler/ops/pool.
    shape_save_template : jinja2.Template
        Generates output dimensions.
        The template is passed from compiler/ops/pool.

    Returns
    -------
    str
        The rendered template of generated function body.

    Raises
    ------
    NotImplementedError
        An error is raised if op_type is not max or average pooling.
    """
    op_type = func_attrs["op"]
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    reduce_op = ""
    if "max" in op_type:
        reduce_op = "ck::ReduceTensorOp::MAX"
    elif "avg" in op_type:
        reduce_op = "ck::ReduceTensorOp::AVG"
    else:
        raise NotImplementedError
    instances = {}
    instance_decl = ""
    for key, _ in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        inst = INSTANCE_TEMPLATE.render(name=fname, reduce_func=reduce_op)
        instances[key] = inst
        instance_decl += inst
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        kernel_h="kernel_h",
        kernel_w="kernel_w",
        stride="stride",
        pad="pad",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
        y_dim3="*in_ch",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""
    for key in instances:
        fname = "f" + sha1(key.encode()).hexdigest()
        program = EXEC_TEMPLATE.render(indent="    ", instance=fname)
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return SRC_TEMPLATE.render(
        instances=instance_decl,
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
    )


def gen_function_decl(func_name):
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


def gen_function_call(func_attrs, indent="  "):
    """
    Parameters
    ----------
    func_attrs : Dict
        Stores the operation attributes.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".

    Returns
    -------
    str
        The rendered template of generated function call.
    """
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_ch="&" + xshape[3]._attrs["name"],
        p_in_h="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_h="&" + yshape[1]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        kernel_h=func_attrs["kernel_size"],
        kernel_w=func_attrs["kernel_size"],
        stride=func_attrs["stride"],
        pad=func_attrs["pad"],
        indent=indent,
    )
