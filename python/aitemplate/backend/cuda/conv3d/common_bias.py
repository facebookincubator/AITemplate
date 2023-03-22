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
CUDA conv3d common functions
"""
import re
from hashlib import sha1
from typing import List

import jinja2

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.conv2d.common import (
    extract_config as conv2d_extract_config,
)
from aitemplate.backend.cuda.gemm_universal.common import (  # noqa: F401
    add_profiler,
    build_profiler,
)

from aitemplate.utils import alignment


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  {% if has_bias %}
  void*,
  {% endif %}
  void*,
  int64_t*, // kernel size
  int64_t*,
  int64_t*,
  int, // strides
  int,
  int,
  int, // padding
  int,
  int,
  int, // dilation
  int,
  int,
  int64_t*, // in_batch
  int64_t*, // in_ch
  int64_t*, // in_t
  int64_t*, // in_h
  int64_t*, // in_w
  int64_t*, // out_ch
  int64_t*, // out_t
  int64_t*, // out_h
  int64_t*, // out_w
  cudaStream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{weight_ptr}},
{% if has_bias %}
{{indent}}    {{bias_ptr}},
{% endif %}
{{indent}}    {{out_ptr}},
{{indent}}    {{p_kernel_t}},
{{indent}}    {{p_kernel_h}},
{{indent}}    {{p_kernel_w}},
{{indent}}    {{stride_t}},
{{indent}}    {{stride_h}},
{{indent}}    {{stride_w}},
{{indent}}    {{padding_t}},
{{indent}}    {{padding_h}},
{{indent}}    {{padding_w}},
{{indent}}    {{dilation_t}},
{{indent}}    {{dilation_h}},
{{indent}}    {{dilation_w}},
{{indent}}    {{p_in_batch}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_in_t}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_out_ch}},
{{indent}}    {{p_out_t}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    stream
{{indent}});
"""
)


def gen_function_decl(func_name):
    return FUNC_DECL_TEMPLATE.render(func_name=func_name, has_bias=True)


def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    w = func_attrs["inputs"][1]
    wshape = w._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    b = func_attrs["inputs"][2]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        weight_ptr=w._attrs["name"],
        has_bias=True,
        bias_ptr=b._attrs["name"],
        out_ptr=y._attrs["name"],
        p_in_batch="&" + xshape[0]._attrs["name"],
        p_out_ch="&" + wshape[0]._attrs["name"],
        p_in_ch="&" + xshape[4]._attrs["name"],
        p_kernel_t="&" + wshape[1]._attrs["name"],
        p_kernel_h="&" + wshape[2]._attrs["name"],
        p_kernel_w="&" + wshape[3]._attrs["name"],
        p_in_t="&" + xshape[1]._attrs["name"],
        p_in_h="&" + xshape[2]._attrs["name"],
        p_in_w="&" + xshape[3]._attrs["name"],
        p_out_t="&" + yshape[1]._attrs["name"],
        p_out_h="&" + yshape[2]._attrs["name"],
        p_out_w="&" + yshape[3]._attrs["name"],
        stride_t=func_attrs["stride"][0],
        stride_h=func_attrs["stride"][1],
        stride_w=func_attrs["stride"][2],
        padding_t=func_attrs["pad"][0],
        padding_h=func_attrs["pad"][1],
        padding_w=func_attrs["pad"][2],
        dilation_t=func_attrs["dilate"][0],
        dilation_h=func_attrs["dilate"][1],
        dilation_w=func_attrs["dilate"][2],
        indent=indent,
    )


KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass{{opcode_class}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)


def emit_instance(op):
    """emit instance"""
    import cutlass_lib

    emiter = cutlass_lib.conv3d_operation.EmitConv3dInstance()
    op_def = emiter.emit(op)
    return op_def


def extract_config(func_attrs, dtype="float16"):
    """Extracts cutlass config for conv kernels."""
    import cutlass_lib

    return conv2d_extract_config(
        func_attrs=func_attrs,
        dtype=dtype,
        op_kind=cutlass_lib.library.OperationKind.Conv3d,
        op_layout="ndhwc",
    )


def extract_config_name(config):
    """Extracts config name from a given config."""
    pattern = re.compile(r"\s*using\s(.*?)\s=")
    decl = config.split("\n")[2]
    match = pattern.match(decl)
    if match is None:
        raise RuntimeError("Invalid config: \n" + config)
    return match.groups()[0]


def gen_function(
    func_attrs,
    instance_template,
    exec_template,
    src_template,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
    f_emit_instance=emit_instance,
    extra_header="",
):
    """Function definition codegen."""
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for key, value in exec_path.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        if value not in inst_def_flag:
            config = f_emit_instance(op_instance[value])
            inst_def_flag.add(value)
        else:
            continue
        inst = instance_template.render(
            config=config, name=fname, config_name=extract_config_name(config)
        )
        instances[key] = inst
        instance_decl += inst
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_d",
        x_dim2="*in_h",
        x_dim3="*in_w",
        x_dim4="*in_ch",
        w_dim0="*out_ch",
        w_dim1="*kernel_d",
        w_dim2="*kernel_h",
        w_dim3="*kernel_w",
        stride_d="stride_d",
        stride_h="stride_h",
        stride_w="stride_w",
        dilate_d="dilation_d",
        dilate_h="dilation_h",
        dilate_w="dilation_w",
        pad_d="pad_d",
        pad_h="pad_h",
        pad_w="pad_w",
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_d",
        y_dim2="*out_h",
        y_dim3="*out_w",
        y_dim4="*out_ch",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""
    for key in instances:
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_template.render(indent="    ", instance=fname, dtype=dtype)
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return src_template.render(
        instances=instance_decl,
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
        extra_header=extra_header,
    )


def cal_align_ab(x_shape: List[int], dtype="float16") -> int:
    """Returns input alignment."""
    k = x_shape[4]  # CI
    return alignment.find_max_alignment(k, dtype)


def function_filter(cfg, func_attrs, x_shape):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    dtype = func_attrs["inputs"][0]._attrs["dtype"]
    ab_alignment = cal_align_ab(x_shape, dtype=dtype)
    tmp = cfg.split("_")
    align_c = int(tmp[-1])
    align_ab = int(tmp[-2])
    if align_c != func_attrs["epilogue_alignment"]:
        return False
    if align_ab != ab_alignment:
        return False
    return True
