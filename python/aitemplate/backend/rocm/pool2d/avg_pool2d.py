# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] rocm avg_pool2d funcs
"""
from ... import registry
from . import pool2d


@registry.reg("rocm.avg_pool2d.gen_function")
def max_pool2d_gen_function(
    func_attrs,
    exec_cond_remplate,
    shape_eval_template,
    shape_save_template,
):
    return pool2d.gen_function(
        func_attrs,
        exec_cond_remplate,
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
