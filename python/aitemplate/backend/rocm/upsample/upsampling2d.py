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
ROCM codegen functions for unsampling2d_add.
"""

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import ROCMSpec
from aitemplate.backend.common import upsampling2d_common

# pylint: disable=C0103,C0415,W0613,C0301,W0612


Header_Files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "library/include/ck/library/utility/host_tensor.hpp"
"""


@registry.reg("rocm.upsampling2d.gen_function")
def gen_function(
    func_attrs,
    template_path,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    backend_spec = ROCMSpec()
    input_type = backend_spec.dtype_to_lib_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_lib_type(y._attrs["dtype"])
    half2_data_ref = backend_spec.half2_data_ref

    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_h",
        x_dim2="*in_w",
        x_dim3="*in_ch",
        scale_factor=func_attrs["scale_factor"],
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_h",
        y_dim2="*out_w",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""
    for key in exec_path:
        program = upsampling2d_common.EXEC_TEMPLATE.render(dtype=input_type)
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return upsampling2d_common.SRC_TEMPLATE.render(
        header_files=Header_Files,
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        elem_input_type=input_type,
        elem_output_type=output_type,
        half2_data_ref=half2_data_ref,
        mode=func_attrs["mode"],
        tsize=upsampling2d_common.gen_alignment(x),
    )


@registry.reg("rocm.upsampling2d.func_decl")
def upsampling2d_gen_function_decl(func_attrs):
    return upsampling2d_common.gen_function_decl(func_attrs, backend_spec=ROCMSpec())


@registry.reg("rocm.upsampling2d.func_call")
def upsampling2d_gen_function_call(func_attrs, indent="    "):
    return upsampling2d_common.gen_function_call(
        func_attrs, backend_spec=ROCMSpec(), indent=indent
    )
