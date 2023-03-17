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
Common codegen functions for ROCM.
"""
import jinja2

from aitemplate.backend import registry

# pylint: disable=W0613

VAR_TEMPLATE = jinja2.Template("""{{indent}} int64_t {{name}} { {{value}} };""")

PTR_TEMPLATE = jinja2.Template("""{{indent}} void * {{name}} {nullptr};""")


@registry.reg("rocm.lib.var_decl")
def var_decl(name, value=0, indent="  "):
    return VAR_TEMPLATE.render(name=name, value=value, indent=indent)


@registry.reg("rocm.lib.void_ptr_decl")
def void_ptr_decl(name, dtype="float16", indent="  "):
    # FIXME: we should just print void* after we support general tensor type, e.g.
    # return PTR_TEMPLATE.render(name=name, dtype="void*", indent=indent)
    if dtype == "float16":
        type_string = "ck::half_t*"
    elif dtype == "int64":
        type_string = "int64_t*"
    elif dtype == "bool":
        type_string = "bool*"
    else:
        raise NotImplementedError
    return PTR_TEMPLATE.render(name=name, dtype=type_string, indent=indent)
