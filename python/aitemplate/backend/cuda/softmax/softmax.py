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
Softmax codegen for CUDA.
"""

import os
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntImm

# pylint: disable=C0301, C0116


# input size: [M, K]
# We put if else condition here to avoid long compilation time.
# i.e. for each K, we only need to compile one of the implementation, not all.
#
# For each K, whether to use wrapReduce or blockReduce was done by experiment
# Please refer to this post: https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F
# and this experiment log [fb internal only]: https://docs.google.com/spreadsheets/d/1bl3GCLQ67p27kXOSVJikEob38fojqaZIS--mPdQxeo0/edit#gid=931264442
FUNC_TEMPLATE = jinja2.Template(
    """
{{custom_libs}}

{{func_signature}}
{
  {{shape_functions}}
  size_t m0 = {{m}};
  size_t n = {{K}};
  size_t m = M;
  bool success = true;

  // For threshold K, please refer to this post: https://fb.quip.com/HCfIAbpWB0qi
  {% if K <= 32 and K % 4 == 0 or K <= 8 %}
    // K <= 32 and K % 4 == 0 or K <= 8
    LaunchSoftmaxSmallK<{{dtype}}, {{K}}, {{m}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
  {% elif K % 8 == 0 %}
    // K % 8 == 0: vector8 kernels
    {% if K/8 <=32 %}
      // K/8 <= 32
      LaunchSoftmaxK8Small<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K <= 3840 %}
      // 32 < K/8 <= 480
      LaunchSoftmaxK8Middle<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K > 3840 %}
      // K/8 > 480
      using vec8 = VecTFor<{{dtype}}>::vec8;
      LaunchSoftmaxBlockAll<vec8, {{dtype}},{{K}}>(reinterpret_cast<const vec8*>(input), reinterpret_cast<vec8*>(output), M, stream, &success);
    {% endif %}
  {% elif K % 4 == 0 %}
    // K % 4 == 0: vector4 kernels
    {% if K/4 <=32 %}
      // K/4 <= 32
      LaunchSoftmaxK4Small<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K <= 1920 %}
      // 32 < K/4 <= 480
      LaunchSoftmaxK4Middle<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K > 1920 %}
      // K/4 > 480
      using vec4 = VecTFor<{{dtype}}>::vec4;
      LaunchSoftmaxBlockAll<vec4,{{dtype}},{{K}}>(reinterpret_cast<const vec4*>(input), reinterpret_cast<vec4*>(output), M, stream, &success);
    {% endif %}
  {% elif K % 2 == 0 %}
    // K % 2 == 0: vector2 kernels
    {% if K/2 <=32 %}
      // K/2 <= 32
      LaunchSoftmaxK2Small<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K <= 1152 %}
      // 32 < K/2 <= 576
      LaunchSoftmaxK2Middle<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K > 1152 %}
      // K/2 > 576
      using vec2 = VecTFor<{{dtype}}>::vec2;
      LaunchSoftmaxBlockAll<vec2,{{dtype}},{{K}}>(reinterpret_cast<const vec2*>(input), reinterpret_cast<vec2*>(output), M, stream, &success);
    {% endif %}
  {% else %}
    // odd K
    {% if K <= 32 %}
      // K <= 32
      LaunchSoftmaxK1Small<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K <= 1408 %}
      // 32 < K <= 1408
      LaunchSoftmaxK1Middle<{{dtype}}, {{K}}>(static_cast<const {{dtype}}*>(input), static_cast<{{dtype}}*>(output), M, stream);
    {% elif K > 1408 %}
      // K > 1408
      LaunchSoftmaxBlockAll<{{dtype}},{{dtype}},{{K}}>( (const {{dtype}}*) input, ({{dtype}}*) output, m, stream, &success);
    {% endif %}
  {% endif %}

  if (!success) {
    softmaxBlockNocache<{{dtype}}><<<m, 1024, 0, stream>>>(({{dtype}}*)input, ({{dtype}}*)output, m, n);
  }
}
    """
)

SHAPE_FUNCTIONS = jinja2.Template(
    """
    int64_t M = 1;
{% for idx in range(input_ndim - 1) %}
    M *= *in_{{idx}};
{% endfor %}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* input,
                   void* output,
{% for idx in range(input_ndim - 1) %}
                   int64_t* in_{{idx}},
{% endfor %}
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
{{indent}}   {{input}},
{{indent}}   {{output}},
{% for name in input_dim_names[:-1] %}
{{indent}}    &{{name}},
{% endfor %}
{{indent}}   stream
{{indent}});
    """
)


def get_func_signature(func_attrs: Dict[str, Any]) -> str:
    input_ndim = func_attrs["inputs"][0]._rank()
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        input_ndim=input_ndim,
    ).strip()


def find_tile_size(k: int) -> int:
    """
    Find the smalled m that would make m * k multiples of 8.
    For odd k, only apply for k <= 5
    """
    m = 1 if k > 1 else 8
    for i in (1, 2, 4, 8):
        if (k * i) % 8 == 0:
            m = i
            break
    if k % 2 == 1 and k >= 7:
        m = 1
    return m


@registry.reg("cuda.softmax.gen_function")
def softmax_gen_function(func_attrs: Dict[str, Any]) -> str:
    dim = func_attrs["dim"]
    shapes = func_attrs["inputs"][0]._attrs["shape"]
    rank = len(shapes)

    assert (
        dim == rank - 1
    ), f"softmax only supports dim == rank - 1, dim={dim}, rank={rank}"

    assert isinstance(
        shapes[dim], IntImm
    ), "softmax requires reduction dim to be static"

    k = shapes[dim].value()

    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    return FUNC_TEMPLATE.render(
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "softmax.cuh"
        ),
        func_signature=get_func_signature(func_attrs),
        shape_functions=SHAPE_FUNCTIONS.render(input_ndim=rank),
        dtype=elem_input_type,
        K=k,
        m=find_tile_size(k),
    )


@registry.reg("cuda.softmax.func_decl")
def softmax_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(func_signature=get_func_signature(func_attrs))


@registry.reg("cuda.softmax.func_call")
def softmax_gen_function_call(func_attrs, indent="  "):
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 1

    input_name = func_attrs["inputs"][0]._attrs["name"]
    output_name = func_attrs["outputs"][0]._attrs["name"]

    shapes = func_attrs["inputs"][0]._attrs["shape"]
    assert (
        len(shapes) >= 2
    ), f"Softmax only supports input with rank >= 2, current rank: {len(shapes)}"

    input_dim_names = [shape._attrs["name"] for shape in shapes]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        output=output_name,
        input_dim_names=input_dim_names,
        indent=indent,
    )
