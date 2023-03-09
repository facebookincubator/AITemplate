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
batched_nms kernel codegen for CUDA.
"""

import os
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec

# pylint: disable=C0301

FUNC_CALL_INT64_PARAM_TEMPLATE = jinja2.Template("reinterpret_cast<int64_t*>({{name}})")

FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include <cub/cub.cuh>

namespace {

{{custom_libs}}

}  // namespace

{{func_signature}}
{

    batched_nms_launcher<{{elem_input_type}}>(
        0, instance_num, keep_n, iou_threshold, input, workspace, output, mask);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(int64_t* output,
                   const void* input,
                   const int instance_num,
                   const int keep_n,
                   const float iou_threshold,
                   int64_t* mask,
                   uint8_t* workspace,
                   cudaStream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{input}},
{{indent}}    {{instance_num}},
{{indent}}    {{keep_n}},
{{indent}}    {{iou_threshold}},
{{indent}}    {{mask}},
{{indent}}    global_workspace_, stream /* default stream */
{{indent}});
    """
)


def get_custom_libs() -> str:
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "batched_nms_kernel.cuh")
    with open(filename) as f:
        res = f.read()
        return res


@registry.reg("cuda.batched_nms.gen_function")
def batched_nms_gen_function(func_attrs: Dict[str, Any]) -> str:
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    return FUNC_TEMPLATE.render(
        elem_input_type=elem_input_type,
        custom_libs=get_custom_libs(),
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
    )


@registry.reg("cuda.batched_nms.func_decl")
def batched_nms_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]).strip()
    )


@registry.reg("cuda.batched_nms.func_call")
def batched_nms_gen_function_call(func_attrs, indent="  "):
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 2

    output_name = FUNC_CALL_INT64_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )
    input_name = func_attrs["inputs"][0]._attrs["name"]
    tmp_name = FUNC_CALL_INT64_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][1]._attrs["name"]
    )

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    instance_num = xshape[0]._attrs["values"][0]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        instance_num=instance_num,
        keep_n=func_attrs["keep_n"],
        iou_threshold=func_attrs["iou_threshold"],
        mask=tmp_name,
        indent=indent,
    )
