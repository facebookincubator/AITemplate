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
ROCM avg_pool2d funcs
"""
from aitemplate.backend import registry
from aitemplate.backend.rocm.pool2d import pool2d


@registry.reg("rocm.avg_pool2d.gen_function")
def max_pool2d_gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    return pool2d.gen_function(
        func_attrs,
        exec_cond_template,
        shape_eval_template,
        shape_save_template,
    )


@registry.reg("rocm.avg_pool2d.func_decl")
def avg_pool2d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return pool2d.gen_function_decl(func_name)


@registry.reg("rocm.avg_pool2d.func_call")
def avg_pool2d_gen_function_call(func_attrs, indent="  "):
    return pool2d.gen_function_call(func_attrs, indent)
