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
Concatenate tanh op for ROCM backend.
"""
import jinja2

from aitemplate.backend import registry
from aitemplate.backend.rocm.tensor import concatenate

TANH_DEF = jinja2.Template(
    """
__device__  half2 fast_tanh(half2 x) {
  // 1-2/(e^(2x)+1)
  const half2 u = __hmul2(half2(2), x);
  const half2 emu = h2exp(u);
  const half2 cdf =
      __hsub2(half2(1), __h2div(half2(2), __hadd2(half2(1), emu)));
  return cdf;
}
__device__  half fast_tanh(half x) {
  // 1-2/(e^(2x)+1)
  const half u = __hmul(half(2), x);
  const half emu = hexp(u);
  const half cdf = __hsub(half(1), __hdiv(half(2), __hadd(half(1), emu)));
  return cdf;
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
"""
)


@registry.reg("rocm.concatenate_tanh.func_decl")
def gen_function_decl(func_attrs):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    Returns
    -------
    str
        Rendered function declaration.
    """
    return concatenate.gen_function_decl(func_attrs)


@registry.reg("rocm.concatenate_tanh.gen_function")
def gen_function(func_attrs):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    element_func_def: str
        Functions that defines how tanh work.

    Returns
    -------
    str
        Rendered function body.
    """
    return concatenate.gen_function(
        func_attrs, element_func="fast_tanh", element_func_def=TANH_DEF.render()
    )


@registry.reg("rocm.concatenate_tanh.func_call")
def gen_function_call(func_attrs, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    return concatenate.gen_function_call(func_attrs, indent=indent)
