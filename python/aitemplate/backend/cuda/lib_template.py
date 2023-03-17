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
Common function templates for CUDA codegen.
"""
import jinja2

from aitemplate.backend import registry

# pylint: disable=C0301

VAR_TEMPLATE = jinja2.Template("""{{indent}} int64_t {{name}} { {{value}} };""")

PTR_TEMPLATE = jinja2.Template("""{{indent}} {{dtype}} {{name}} {nullptr};""")


@registry.reg("cuda.lib.var_decl")
def var_decl(name, value=0, indent="  "):
    return VAR_TEMPLATE.render(name=name, value=value, indent=indent)


@registry.reg("cuda.lib.void_ptr_decl")
def void_ptr_decl(name, dtype="float16", indent="  "):
    # FIXME: we keep dtype in void_ptr_decl's param list because rocm needs it.
    # We will remove it once we support general tensor type for rocm
    return PTR_TEMPLATE.render(name=name, dtype="void*", indent=indent)
