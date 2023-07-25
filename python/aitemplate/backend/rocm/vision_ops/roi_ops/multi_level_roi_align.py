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
ROCM codegen functions for multi-level roi align.
"""

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import ROCMSpec
from aitemplate.backend.common.vision_ops import multi_level_roi_align_common

# pylint: disable=C0103,C0415,W0613,C0301,W0612

EXTRA_HEADER = jinja2.Template(
    """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "library/include/ck/library/utility/host_tensor.hpp"
"""
)


@registry.reg("rocm.multi_level_roi_align.gen_function")
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
    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])
    exec_paths = ""
    for key, _ in exec_path.items():
        program = multi_level_roi_align_common.EXEC_TEMPLATE.render(
            indent="    ",
            num_rois=func_attrs["num_rois"],
            pooled_size=func_attrs["pooled_size"],
            sampling_ratio=func_attrs["sampling_ratio"],
            spatial_scale=func_attrs["spatial_scale"],
            position_sensitive=func_attrs["position_sensitive"],
            continuous_coordinate=func_attrs["continuous_coordinate"],
            elem_input_type=input_type,
            elem_output_type=output_type,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return multi_level_roi_align_common.SRC_TEMPLATE.render(
        function_name=func_name,
        exec_paths=exec_paths,
        header_files=EXTRA_HEADER.render(),
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        elem_input_type=input_type,
        elem_output_type=output_type,
    )


@registry.reg("rocm.multi_level_roi_align.func_decl")
def multi_level_roi_align_gen_function_decl(func_attrs):
    return multi_level_roi_align_common.gen_function_decl(
        func_attrs, backend_spec=ROCMSpec()
    )


@registry.reg("rocm.multi_level_roi_align.func_call")
def multi_level_roi_align_gen_function_call(func_attrs, indent="  "):
    return multi_level_roi_align_common.gen_function_call(
        func_attrs, backend_spec=ROCMSpec()
    )
