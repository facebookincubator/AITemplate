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
from __future__ import annotations

import math
import os
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.target import Target

from aitemplate.compiler.base import IntImm

# pylint: disable=C0301, C0116


# input size: [M, K] where M == outer size (product of outer dims) and K == reduction dim
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
  {{func_impl}}
}
    """
)

FUNC_IMPL_INNER_SIZE_EQ_1 = jinja2.Template(
    """
  const size_t M = outer_size;
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
      LaunchSoftmaxBlockAll<vec8, {{dtype}}, {{K}}>(reinterpret_cast<const vec8*>(input), reinterpret_cast<vec8*>(output), M, stream, &success);
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
      LaunchSoftmaxBlockAll<vec4, {{dtype}}, {{K}}>(reinterpret_cast<const vec4*>(input), reinterpret_cast<vec4*>(output), M, stream, &success);
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
      LaunchSoftmaxBlockAll<vec2, {{dtype}}, {{K}}>(reinterpret_cast<const vec2*>(input), reinterpret_cast<vec2*>(output), M, stream, &success);
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
      LaunchSoftmaxBlockAll<{{dtype}}, {{dtype}}, {{K}}>( (const {{dtype}}*) input, ({{dtype}}*) output, M, stream, &success);
    {% endif %}
  {% endif %}

  if (!success) {
    softmaxBlockNocache<{{dtype}}><<<M, 1024, 0, stream>>>(({{dtype}}*)input, ({{dtype}}*)output, M, {{K}});
  }
    """
)

FUNC_IMPL_GENERAL = jinja2.Template(
    """
    LaunchSoftmaxGeneral<{{dtype}}, {{dim_size}}, {{inner_size}}, {{dim_threads}}, {{inner_threads}}>(
        (const {{dtype}}*) input,
        ({{dtype}}*) output,
        outer_size,
        multiprocessor_count,
        stream
    );
    """
)

SHAPE_FUNCTIONS = jinja2.Template(
    """
    int64_t outer_size = 1;
{% for idx in range(reduction_dim) %}
    outer_size *= *in_{{idx}};
{% endfor %}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* input,
               void* output,
{% for idx in range(reduction_dim) %}
               int64_t* in_{{idx}},
{% endfor %}
               int multiprocessor_count,
               cudaStream_t stream)
    """,
    trim_blocks=True,
)

FUNC_DECL = jinja2.Template(
    """
{{func_signature}};
    """,
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{input}},
{{indent}}   {{output}},
{% for name in outer_dim_names %}
{{indent}}   &{{name}},
{% endfor %}
{{indent}}   device_properties_.multiProcessorCount,
{{indent}}   stream
{{indent}});
    """,
    trim_blocks=True,
)


def get_func_signature(func_attrs: Dict[str, Any]) -> str:
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        reduction_dim=func_attrs["dim"],
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


def _softmax_general_block_size(dim_size: int, inner_size: int) -> tuple[int, int]:
    MAX_THREADS_PER_BLOCK = 1024
    WARP_SIZE = 32
    assert inner_size != 0
    inner_threads = min(inner_size, MAX_THREADS_PER_BLOCK)
    dim_threads = 1
    if inner_threads <= 32 and dim_size >= WARP_SIZE:
        dim_threads = (
            min(
                MAX_THREADS_PER_BLOCK // inner_threads // WARP_SIZE,
                dim_size // WARP_SIZE,
            )
            * WARP_SIZE
        )
    # When dim_threads > 1, warp reduction is done, and for our warp reduction
    # impl to work, dim_threads needs to be a multiple of the warp size.
    assert dim_threads == 1 or dim_threads % WARP_SIZE == 0
    assert dim_threads != 0
    return dim_threads, inner_threads


@registry.reg("cuda.softmax.gen_function")
def softmax_gen_function(func_attrs: Dict[str, Any]) -> str:
    reduction_dim = func_attrs["dim"]
    shape = func_attrs["inputs"][0]._attrs["shape"]

    assert all(
        isinstance(dim, IntImm) for dim in shape[reduction_dim:]
    ), "softmax requires reduction dim & inner dims to be static"

    dim_size = shape[reduction_dim].value()

    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )

    inner_size = math.prod(dim.value() for dim in shape[reduction_dim + 1 :])
    if inner_size == 1:
        func_impl = FUNC_IMPL_INNER_SIZE_EQ_1.render(
            dtype=elem_input_type,
            m=find_tile_size(dim_size),
            K=dim_size,
        )
    else:
        dim_threads, inner_threads = _softmax_general_block_size(dim_size, inner_size)
        func_impl = FUNC_IMPL_GENERAL.render(
            dtype=elem_input_type,
            dim_size=dim_size,
            inner_size=inner_size,
            dim_threads=dim_threads,
            inner_threads=inner_threads,
        )

    return FUNC_TEMPLATE.render(
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "softmax.cuh"
        ),
        func_signature=get_func_signature(func_attrs),
        func_impl=func_impl,
        shape_functions=SHAPE_FUNCTIONS.render(reduction_dim=reduction_dim),
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

    shape = func_attrs["inputs"][0]._attrs["shape"]
    assert (
        len(shape) >= 2
    ), f"Softmax only supports input with rank >= 2, current rank: {len(shape)}"

    reduction_dim = func_attrs["dim"]
    outer_dim_names = [dim._attrs["name"] for dim in shape[:reduction_dim]]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        output=output_name,
        outer_dim_names=outer_dim_names,
        indent=indent,
    )
