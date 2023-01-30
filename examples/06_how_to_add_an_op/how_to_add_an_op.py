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

from typing import Any, Dict, List

import jinja2
import torch

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec, ROCMSpec
from aitemplate.compiler import compile_model
from aitemplate.compiler.base import IntVar, Operator, Tensor
from aitemplate.testing import detect_target


class add_one(Operator):
    def __init__(self):
        super().__init__()
        # required, unique identity of operator category
        self._attrs["op"] = "add_one"
        # we can put whatever we want into the op attrs for later use
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(self, x: Tensor) -> Tensor:
        # each operator needs to keep a record of input tensors
        self._attrs["inputs"] = [x]
        # optional, to set depth of the op based on inputs' depth, used in DFS
        self._set_depth()
        # infer output shape
        output_shape = self._infer_shape(x)
        # create output Tensor, of which the source op is the current op
        output = Tensor(output_shape, src_ops={self})
        # remember current op's outputs
        self._attrs["outputs"] = [output]
        return output

    def _infer_shape(self, x) -> List[IntVar]:
        return x.shape()

    def gen_function(self) -> str:
        # this function will be used in codegen
        # here we only need to redirect to backend codegen function
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)


FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{kernel}}

}  // namespace

{{func_signature}}
{
    invoke_add_one(
        static_cast<{{elem_type}}*>(output),
        static_cast<const {{elem_type}}*>(input),
        num_elements,
        stream);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* input,
                   const int64_t num_elements,
                   {{prefix}}Stream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}int64_t num_elements = 1;
{% for dim_name in dim_names %}
{{indent}}num_elements *= {{dim_name}};
{% endfor %}

{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{input}}, num_elements, stream /* default stream */
{{indent}});
    """
)


KERNEL_TEMPLATE = jinja2.Template(
    """
__global__ void add_one({{elem_type}}* output, const {{elem_type}}* input, const int64_t num_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = input[idx] + {{elem_type}}(1.0);
  }
}

void invoke_add_one({{elem_type}}* output, const {{elem_type}}* input, int64_t num_elements, {{prefix}}Stream_t stream) {
  if (num_elements < 1024) {
    dim3 grid(1);
    dim3 block(num_elements);
    add_one<<<grid, block, 0, stream>>>(output, input, num_elements);
  } else {
    dim3 grid((num_elements + 1024 - 1) / 1024);
    dim3 block(1024);
    add_one<<<grid, block, 0, stream>>>(output, input, num_elements);
  }
}
    """
)


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 1

    output_name = func_attrs["outputs"][0]._attrs["name"]
    input_name = func_attrs["inputs"][0]._attrs["name"]

    dim_names = [dim._attrs["name"] for dim in func_attrs["inputs"][0].shape()]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        dim_names=dim_names,
        indent=indent,
    )


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    input_x = func_attrs["inputs"][0]
    output_y = func_attrs["outputs"][0]
    input_type = backend_spec.dtype_to_backend_type(input_x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(output_y._attrs["dtype"])

    if input_type != output_type:
        raise NotImplementedError("input type must equal to output type")

    prefix = backend_spec.prefix

    return FUNC_TEMPLATE.render(
        header_files=header_files,
        elem_type=input_type,
        kernel=KERNEL_TEMPLATE.render(prefix=prefix, elem_type=input_type),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=prefix
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
        ).strip()
    )


CUDA_HEADER_FILES = """
#include <cuda_fp16.h>
"""


@registry.reg("cuda.add_one.gen_function")
def cuda_add_one_gen_function(func_attrs: Dict[str, Any]) -> str:
    return gen_function(func_attrs, CUDA_HEADER_FILES, CUDASpec())


@registry.reg("cuda.add_one.func_decl")
def cuda_add_one_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.add_one.func_call")
def cuda_add_one_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return gen_function_call(func_attrs, indent)


HIP_HEADER_FILES = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
"""


@registry.reg("rocm.add_one.gen_function")
def rocm_add_one_gen_function(func_attrs: Dict[str, Any]) -> str:
    return gen_function(func_attrs, HIP_HEADER_FILES, ROCMSpec())


@registry.reg("rocm.add_one.func_decl")
def rocm_add_one_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.add_one.func_call")
def rocm_add_one_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return gen_function_call(func_attrs, indent)


def create_ait_model(shapes):
    X = Tensor(
        shape=shapes,
        dtype="float16",
        name="X",
        is_input=True,
    )
    Y = add_one()(X)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    return Y


def verify_add_one():
    shapes = [16, 512]
    x = torch.randn(shapes).cuda().half()
    y_pt = x + 1.0

    Y = create_ait_model([16, 512])
    target = detect_target()
    with compile_model(Y, target, "./tmp", "add_one") as module:
        y = torch.empty(shapes).cuda().half()
        inputs = {"X": x}
        outputs = {"Y": y}
        module.run_with_tensors(inputs, outputs)
        print(torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2))


verify_add_one()
