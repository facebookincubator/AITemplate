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
Common functions for layernorm kernels
"""

from typing import Any, Dict, List

import jinja2

FUNC_CALL_FP16_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<half*>(&({{name}}->raw()))"
)

GAMMA_BETA_CONST_DEFS_TEMPLATE = jinja2.Template(
    """
{% if is_gamma_const %}
#define AIT_LAYERNORM_CONST_GAMMA {{gamma_constant}}
{% endif %}
{% if is_beta_const %}
#define AIT_LAYERNORM_CONST_BETA {{beta_constant}}
{% endif %}

"""
)

SHAPE_PRODUCT_TEMPLATE = jinja2.Template(
    """
{{indent}}int64_t {{M}} = 1;
{% for dim_name in dim_names %}
{{indent}}{{M}} *= {{dim_name}};
{% endfor %}
    """
)


def generate_m_shape_func(input_shapes, norm_ndim, m_name, indent):
    m_names = [shape._attrs["name"] for shape in input_shapes[:-norm_ndim]]
    return SHAPE_PRODUCT_TEMPLATE.render(
        M=m_name,
        dim_names=m_names,
        indent=indent,
    )


def generate_n_shape_func(input_shapes, norm_ndim, n_name, indent):
    n_names = [shape._attrs["name"] for shape in input_shapes[-norm_ndim:]]
    return SHAPE_PRODUCT_TEMPLATE.render(
        M=n_name,
        dim_names=n_names,
        indent=indent,
    )


def gamma_beta_const_defs(func_attrs: Dict[str, Any]) -> str:
    """
    Return rendered code string where we define the default gamma (1.0) and
    beta (0.0) values, respectively.
    """
    gamma_constant = func_attrs["gamma_constant"]
    is_gamma_const = gamma_constant is not None
    beta_constant = func_attrs["beta_constant"]
    is_beta_const = beta_constant is not None

    return GAMMA_BETA_CONST_DEFS_TEMPLATE.render(
        is_gamma_const=is_gamma_const,
        gamma_constant=gamma_constant,
        is_beta_const=is_beta_const,
        beta_constant=beta_constant,
    )


def get_input_names(func_attrs: Dict[str, Any]) -> List[str]:
    """
    Return a list of rendered name strings for inputs. It returns nullptr
    for gamma and beta if they are None.
    """
    inputs = func_attrs["inputs"]
    x = inputs[0]
    gamma = None
    beta = None

    idx = 1
    if func_attrs["gamma_constant"] is None:
        gamma = inputs[idx]
        idx += 1
    if func_attrs["beta_constant"] is None:
        beta = inputs[idx]
        idx += 1

    input_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(name=x._attrs["name"])
    if gamma is None:
        gamma_name = "nullptr"
    else:
        gamma_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(name=gamma._attrs["name"])
    if beta is None:
        beta_name = "nullptr"
    else:
        beta_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(name=beta._attrs["name"])

    return (input_name, gamma_name, beta_name)
