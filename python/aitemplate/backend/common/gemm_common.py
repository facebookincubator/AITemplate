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
Backend-agnostic functions for gemm codegen.
"""

from typing import Dict

import jinja2

from aitemplate.compiler.ops.gemm_universal.gemm_common import DimInfo, Source

SHAPE_EVAL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}} {{name}} = {{dim_calculator}};
"""
)


def gen_dim_calculator(dim_info: DimInfo, is_ptr: bool) -> str:
    prefix = "*" if is_ptr else ""
    if dim_info.source == Source.INPUT:
        if dim_info.tensor_idx == 0:
            prefix += "a_dim"
        else:
            assert dim_info.tensor_idx == 1, f"Unsupported gemm dim: {dim_info}"
            prefix += "b_dim"
    else:
        assert (
            dim_info.source == Source.OUTPUT and dim_info.tensor_idx == 0
        ), f"Unsupported gemm dim: {dim_info}"
        prefix += "c_dim"
    dim_names = ["(" + prefix + str(idx) + ")" for idx in dim_info.dim_idx]
    return " * ".join(dim_names)


def gen_shape_eval_code(
    indent: int, dtype: str, dim_info_dict: Dict[str, DimInfo], is_ptr: bool
) -> str:
    shape_eval_list = []
    for name, dim_info_list in dim_info_dict.items():
        dim_info = None
        for d in dim_info_list:
            if d.placeholder:
                continue

            dim_info = d
            break
        assert dim_info is not None, f"Couldn't find valid dim info for dim {name}"

        shape_eval_list.append(
            SHAPE_EVAL_TEMPLATE.render(
                dtype=dtype,
                indent=" " * indent,
                name=name,
                dim_calculator=gen_dim_calculator(dim_info, is_ptr),
            )
        )
    return "\n".join(shape_eval_list)
