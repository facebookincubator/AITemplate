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
common functions for transposed conv2d
"""

import re

from aitemplate.backend.cuda.conv2d import common


def _conv_transpose_instance(op_def):
    tmp = op_def.replace("DefaultConv2dFprop", "DefaultConv2dDgrad")
    tmp = re.sub(
        r"cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<\d>",
        "cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>",
        tmp,
    )
    return tmp


def emit_instance(op, f_instance_convertor=_conv_transpose_instance):
    import cutlass_lib

    emiter = cutlass_lib.conv2d_operation.EmitConv2dInstance()
    op_def = emiter.emit(op)
    op_def = f_instance_convertor(op_def)
    return op_def


def extract_config(
    func_attrs,
    dtype="float16",
    skip_simt_kernels=False,
    op_kind=None,
    op_layout=None,
):
    def apply_special_config(func_attrs, op):
        import cutlass_lib

        op.group_mode = cutlass_lib.library.GroupMode.NoneGroup
        return op

    return common.extract_config(
        func_attrs,
        dtype,
        skip_simt_kernels,
        f_apply_special_config=apply_special_config,
        op_kind=op_kind,
        op_layout=op_layout,
    )
