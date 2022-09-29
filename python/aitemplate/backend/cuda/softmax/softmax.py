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

from ....compiler.base import IntImm

from ... import registry
from ...target import Target

# pylint: disable=C0301, C0116

FUNC_CALL_FP16_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<cutlass::half_t*>(&({{name}}->raw()))"
)

FUNC_CALL_FP32_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<float*>(&({{name}}->raw()))"
)

# input size: [M, K]
# We put if else condition here to avoid long compilation time.
# i.e. for each K, we only need to compile one of the implementation, not all.
#
# For each K, whether to use wrapReduce or blockReduce was done by experiment
# Please refer to this post: https://fb.quip.com/HCfIAbpWB0qi
# and this experiment log: https://docs.google.com/spreadsheets/d/1bl3GCLQ67p27kXOSVJikEob38fojqaZIS--mPdQxeo0/edit#gid=931264442
FUNC_TEMPLATE = jinja2.Template(
    """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/platform/platform.h"
#include <math_constants.h>
#include <assert.h>
#include <cuda.h>
namespace {

{{custom_libs}}

}  // namespace

{{func_signature}}
{
  {{shape_functions}}
  size_t m0 = {{m}};
  size_t n = {{K}};
  size_t m = M;
  bool success = true;

  {% if K <= 32 and K % 4 == 0 or K <= 8 %}
    const int n_threads = 128;
    const int m0_by_n_threads = m0 * n_threads;
    dim3 block(n_threads);
    dim3 grid((m + m0_by_n_threads - 1) / m0_by_n_threads);
    softmax_small_k<{{dtype}}, float4, n_threads, {{K}}, {{m}}>
        <<<grid, block, 0, stream>>>({input, output}, m);
  {% elif K % 8 == 0 %}
    {% if K/8 <=32 %}
      int thread_group_width = -1;
      for(auto i: {1, 8, 16, 32}){
        if (8*i >= n){
          thread_group_width = i;
          break;
        }
      }
      int thread_group_per_block = 128/thread_group_width;
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(thread_group_width, thread_group_per_block);
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float8,{{dtype}},8><<<grid, block, 0, stream>>>( (const float8*)input, (float8*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<float4,{{dtype}},8><<<grid, block, 0, stream>>>( (const float4*)input, (float4*)output, m, n);
      {% endif %}
    {% elif K <= 3840 %} // For threshold K, please refer to this post: https://fb.quip.com/HCfIAbpWB0qi
      int thread_group_per_block = 128/32;//4
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(32,thread_group_per_block);
      const int num_packs = (int(({{K}}+31)/32)+7)/8;
      const int cols_per_thread = num_packs * 8;
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float8,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const float8*)input, (float8*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<float4,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const float4*)input, (float4*)output, m, n);
      {% endif %}
    {% elif dtype=="float" and K > 3840 %}
        LaunchSoftmaxBlockAll<float8,{{dtype}},{{K}}>( (const float8*) input, (float8*) output, m, stream, &success);
    {% elif "half" in dtype and K > 3840 %}
        LaunchSoftmaxBlockAll<float4,{{dtype}},{{K}}>( (const float4*) input, (float4*) output, m, stream, &success);
    {% endif %}
  {% elif K % 4 == 0 %}
    {% if K/4 <=32 %}
      int thread_group_width = -1;
      for(auto i: {1, 4, 8, 16, 32}){
        if (4*i >= n){
          thread_group_width = i;
          break;
        }
      }
      int thread_group_per_block = 128/thread_group_width;
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(thread_group_width, thread_group_per_block);
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float4,{{dtype}},8><<<grid, block, 0, stream>>>( (const float4*)input, (float4*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<float2,{{dtype}},8><<<grid, block, 0, stream>>>( (const float2*)input, (float2*)output, m, n);
      {% endif %}
    {% elif K <= 1920 %} // For threshold K, please refer to this post: https://fb.quip.com/HCfIAbpWB0qi
      int thread_group_per_block = 128/32;//4
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(32,thread_group_per_block);
      const int num_packs = (int(({{K}}+31)/32)+3)/4;
      const int cols_per_thread = num_packs * 8;
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float4,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const float4*)input, (float4*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<float2,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const float2*)input, (float2*)output, m, n);
      {% endif %}
    {% elif dtype=="float" and K > 1920 %}
        LaunchSoftmaxBlockAll<float4,{{dtype}},{{K}}>( (const float4*) input, (float4*) output, m, stream, &success);
    {% elif "half" in dtype and K > 1920 %}
        LaunchSoftmaxBlockAll<float2,{{dtype}},{{K}}>( (const float2*) input, (float2*) output, m, stream, &success);
    {% endif %}
  {% elif K % 2 == 0 %}
    {% if K/2 <=32 %}
      int thread_group_width = -1;
      for(auto i: {1, 2, 4, 8, 16, 32}){
        if (2*i >= n){
          thread_group_width = i;
          break;
        }
      }
      int thread_group_per_block = 128/thread_group_width;
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(thread_group_width, thread_group_per_block);
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float2,{{dtype}},8><<<grid, block, 0, stream>>>( (const float2*)input, (float2*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<float,{{dtype}},8><<<grid, block, 0, stream>>>( (const float*)input, (float*)output, m, n);
      {% endif %}
    {% elif K <= 1152 %} // For threshold K, please refer to this post: https://fb.quip.com/HCfIAbpWB0qi
      int thread_group_per_block = 128/32;//4
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(32,thread_group_per_block);
      const int num_packs = (int(({{K}}+31)/32)+1)/2;
      const int cols_per_thread = num_packs * 2;
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float2,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const float2*)input, (float2*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<float,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const float*)input, (float*)output, m, n);
      {% endif %}
    {% elif dtype=="float" and K > 1152 %}
        LaunchSoftmaxBlockAll<float2,{{dtype}},{{K}}>( (const float2*) input, (float2*) output, m, stream, &success);
    {% elif "half" in dtype and K > 1152 %}
        LaunchSoftmaxBlockAll<float,{{dtype}},{{K}}>( (const float*) input, (float*) output, m, stream, &success);
    {% endif %}
  {% else %}
    {% if K <=32 %}
      int thread_group_width = -1;
      for(auto i: {1, 2, 4, 8, 16, 32}){
        if (i >= n){
          thread_group_width = i;
          break;
        }
      }
      int thread_group_per_block = 128/thread_group_width;
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(thread_group_width, thread_group_per_block);
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float,{{dtype}},8><<<grid, block, 0, stream>>>( (const float*)input, (float*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<half,{{dtype}},8><<<grid, block, 0, stream>>>( (const half*)input, (half*)output, m, n);
      {% endif %}
    {% elif K <= 1408 %} // For threshold K, please refer to this post: https://fb.quip.com/HCfIAbpWB0qi
      int thread_group_per_block = 128/32;//4
      int grid_dim_x = (m+thread_group_per_block-1)/thread_group_per_block;
      dim3 grid(grid_dim_x);
      dim3 block(32,thread_group_per_block);
      const int cols_per_thread = ({{K}}+31)/32;
      {% if dtype=="float" %}
        softmax_stored_locally_multi_dim<float,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const float*)input, (float*)output, m, n);
      {% elif "half" in dtype %}
        softmax_stored_locally_multi_dim<half,{{dtype}},cols_per_thread><<<grid, block, 0, stream>>>((const half*)input, (half*)output, m, n);
      {% endif %}
    {% elif dtype=="float" and K > 1408 %}
        LaunchSoftmaxBlockAll<float,{{dtype}},{{K}}>( (const float*) input, (float*) output, m, stream, &success);
    {% elif "half" in dtype and K > 1408 %}
        LaunchSoftmaxBlockAll<half,{{dtype}},{{K}}>( (const half*) input, (half*) output, m, stream, &success);
    {% endif %}
  {% endif %}

  if(!success){
    softmaxBlockNocache<half><<<m, 1024, 0, stream>>>((half*)input, (half*)output, m, n);
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
void {{func_name}}({{dtype}}* input,
                   {{dtype}}* output,
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
        dtype="cutlass::half_t",
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

    return FUNC_TEMPLATE.render(
        custom_libs=Target.current().get_custom_libs(
            os.path.dirname(__file__), "softmax.cuh"
        ),
        func_signature=get_func_signature(func_attrs),
        shape_functions=SHAPE_FUNCTIONS.render(input_ndim=rank),
        dtype="cutlass::half_t",
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

    input_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][0]._attrs["name"]
    )
    output_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"]
    )

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
