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
IntElementwise codegen for CUDA.
"""

import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CPUBackendSpec

from aitemplate.compiler.base import IntVarTensor


INT_VAR_FUNC_TEMPLATE = jinja2.Template(
    """
      {{lhs}} = {{rhs}};
"""
)


@registry.reg("cuda.int_elementwise.gen_function")
def dummpy_int_elementwise_gen_function(func_attrs):
    return ""


@registry.reg("cuda.int_elementwise.func_decl")
def dummpy_int_elementwise_gen_function_decl(func_attrs):
    return ""


@registry.reg("cuda.int_elementwise.func_call")
def int_elementwise_gen_function_call(func_attrs, indent):
    """Generates int_elementwise function call."""
    func_enum = func_attrs["func"]
    inputs = func_attrs["inputs"]
    outputs = func_attrs["outputs"]
    assert (
        len(outputs) == 1
    ), f"Elementwise op for IntVarTensor should only generate 1 output, got {len(outputs)}"
    input_params_vec = []
    for inp in inputs:
        assert isinstance(
            inp, IntVarTensor
        ), f"only inputs of IntVarTensor are allowed for OP with output of IntVarTensor, got type{inp}"
        input_params_vec.append(inp._attrs["int_var"]._attrs["name"])
    backend_spec = CPUBackendSpec()
    op = backend_spec.func_enum_to_func_name.get(func_enum)
    rhs = op.join(input_params_vec)
    lhs = outputs[0]._attrs["name"]
    func_call = INT_VAR_FUNC_TEMPLATE.render(
        lhs=lhs,
        rhs=rhs,
    )
    return func_call
