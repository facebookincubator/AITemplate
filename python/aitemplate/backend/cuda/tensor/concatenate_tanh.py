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
Codegen functions for concatenate_tanh.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.tensor import concatenate


TANH_DEF = jinja2.Template(
    """
#include <cutlass/fast_math.h>

{% if dtype == "half" %}
#ifndef __HALF2_TO_UI
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#endif

#ifndef __HALF_TO_US
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#endif

__device__  half2 fast_tanh(half2 x) {
  #if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)

  asm volatile ( "tanh.approx.f16x2 %0, %1;" : "=r"(__HALF2_TO_UI(x)) : "r"(__HALF2_TO_UI(x)));
  return x;

  #else
  CUTLASS_NOT_IMPLEMENTED();
  #endif
}

__device__  half fast_tanh(half x) {
  #if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)

  asm volatile ( "tanh.approx.f16 %0, %1;" : "=h"(__HALF_TO_US(x)) : "h"(__HALF_TO_US(x)));
  return x;

  #else
  return half(cutlass::fast_tanh(float(x)));
  #endif
}

__device__  float fast_tanh(float x) {
    float y;
    half2* x_vec = (half2*)(&x);
    half2* y_vec = (half2*)(&y);
    y_vec[0] =  fast_tanh(x_vec[0]);
    return y;
}

__device__  float2 fast_tanh(float2 x) {
    float2 y;
    half2* x_vec = (half2*)(&x);
    half2* y_vec = (half2*)(&y);
    y_vec[0] = fast_tanh(x_vec[0]);
    y_vec[1] = fast_tanh(x_vec[1]);
    return y;
}

__device__  float4 fast_tanh(float4 x) {
    float4 y;
    half2* x_vec = (half2*)(&x);
    half2* y_vec = (half2*)(&y);
    y_vec[0] = fast_tanh(x_vec[0]);
    y_vec[1] = fast_tanh(x_vec[1]);
    y_vec[2] = fast_tanh(x_vec[2]);
    y_vec[3] = fast_tanh(x_vec[3]);
    return y;
}
{% elif dtype == "float" %}
__device__  float fast_tanh(float x) {
    return cutlass::fast_tanh(x);
}

__device__  float2 fast_tanh(float2 x) {
    float2 y;
    y.x = cutlass::fast_tanh(x.x);
    y.y = cutlass::fast_tanh(x.y);
    return y;
}

__device__  float4 fast_tanh(float4 x) {
    float4 y;
    y.x = cutlass::fast_tanh(x.x);
    y.y = cutlass::fast_tanh(x.y);
    y.z = cutlass::fast_tanh(x.z);
    y.w = cutlass::fast_tanh(x.w);
    return y;
}
{% endif %}
"""
)


@registry.reg("cuda.concatenate_tanh.func_decl")
def gen_function_decl(func_attrs):
    return concatenate.gen_function_decl(func_attrs)


@registry.reg("cuda.concatenate_tanh.gen_function")
def gen_function(func_attrs):
    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"],
    )

    return concatenate.gen_function(
        func_attrs,
        element_func="fast_tanh",
        element_func_def=TANH_DEF.render(
            dtype=dtype,
        ),
    )


@registry.reg("cuda.concatenate_tanh.func_call")
def gen_function_call(func_attrs, indent="  "):
    return concatenate.gen_function_call(func_attrs, indent=indent)
