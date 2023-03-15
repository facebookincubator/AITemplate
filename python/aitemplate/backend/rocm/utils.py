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
Util functions for ROCM.
"""
import os
import pathlib
import re
import shutil
import tempfile

from aitemplate.backend import registry

# from . import extra_conv_emit, extra_cutlass_generator, extra_enum

# pylint: disable=C0103,C0415,W0707


class Args:
    def __init__(self, arch):
        self.operations = "all"
        self.build_dir = ""
        self.curr_build_dir = ""
        self.rocm_version = "5.0.2"
        self.generator_target = ""
        self.architectures = arch
        self.kernels = "all"
        self.ignore_kernels = ""
        self.kernel_filter_file = None
        self.selected_kernel_list = None
        self.interface_dir = None
        self.filter_by_cc = True


@registry.reg("rocm.make_ck_lib")
def mk_ck_lib(src_prefix, dst_prefix=None):
    if dst_prefix is None:
        dst_prefix = tempfile.mkdtemp()
    lib_dst = os.path.join(dst_prefix, "ck_lib")
    if pathlib.Path(lib_dst).is_dir():
        shutil.rmtree(lib_dst)

    os.makedirs(lib_dst)
    with open(os.path.join(lib_dst, "__init__.py"), "w") as fo:
        fo.write("from . import library\n")
        fo.write("from . import generator\n")
        fo.write("from . import manifest\n")
        fo.write("from . import gemm_operation\n")
        fo.write("from . import conv2d_operation\n")

    def process_code(src_path, dst_path, code_set):
        pattern = re.compile(r"from\s([a-z_0-9]+)\simport \*")
        with open(src_path) as fi:
            lines = fi.readlines()
        output = []

        for line in lines:
            match = pattern.match(line)
            if match is not None:
                name = match.groups()[0]
                if name + ".py" in code_set:
                    line = "from .{name} import *\n".format(name=name)
            output.append(line)
        # if "library.py" in dst_path:
        #     lines = extra_enum.emit_library()
        #     output.append(lines)
        # if "conv2d_operation.py" in dst_path:
        #     lines = extra_conv_emit.emit_library()
        #     output.append(lines)
        with open(dst_path, "w") as fo:
            fo.writelines(output)

    srcs = os.listdir(src_prefix)
    for file in srcs:
        src_path = os.path.join(src_prefix, file)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(lib_dst, file)
        process_code(src_path, dst_path, srcs)

    # extra configs
    # dst_path = os.path.join(lib_dst, "extra_operation.py")
    # with open(dst_path, "w") as fo:
    #     code = extra_ck_generator.emit_library()
    #     fo.write(code)
    return dst_prefix


@registry.reg("rocm.gen_ck_ops")
def gen_ops(arch):
    import ck_lib

    args = Args(arch)
    manifest = ck_lib.manifest.Manifest(args)
    try:
        func = getattr(ck_lib.generator, "Generate" + arch.upper())
        func(manifest, args.rocm_version)
    except AttributeError as exc:
        raise NotImplementedError(
            "Arch " + arch + " is not supported by current cklib lib."
        ) from exc
    return manifest.operations
