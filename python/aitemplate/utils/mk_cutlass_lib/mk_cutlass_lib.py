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
import os
import pathlib
import re
import shutil
import tempfile

from aitemplate.utils.mk_cutlass_lib import (
    extra_conv_emit,
    extra_cutlass_generator,
    extra_enum,
    extra_gemm_emit,
)


def mk_cutlass_lib(template_path, dst_prefix=None):
    if dst_prefix is None:
        dst_prefix = tempfile.mkdtemp()
    lib_dst = os.path.join(dst_prefix, "cutlass_lib")
    if pathlib.Path(lib_dst).is_dir():
        shutil.rmtree(lib_dst)

    os.makedirs(lib_dst)
    with open(os.path.join(lib_dst, "__init__.py"), "w") as fo:
        fo.write("from . import library\n")
        fo.write("from . import generator\n")
        fo.write("from . import manifest\n")
        fo.write("from . import conv3d_operation\n")
        fo.write("from . import gemm_operation\n")
        fo.write("from . import conv2d_operation\n")
        fo.write("from . import extra_operation\n")

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
        if "library.py" in dst_path:
            lines = extra_enum.emit_library()
            output.append(lines)
        if "conv2d_operation.py" in dst_path:
            lines = extra_conv_emit.emit_library()
            output.append(lines)
        if "gemm_operation.py" in dst_path:
            lines = extra_gemm_emit.emit_library()
            output.append(lines)
        with open(dst_path, "w") as fo:
            fo.writelines(output)

    src_prefix = os.path.join(template_path, "tools/library/scripts")
    srcs = os.listdir(src_prefix)
    if "__init__.py" in srcs:
        srcs.remove("__init__.py")
    for file in srcs:
        src_path = os.path.join(src_prefix, file)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(lib_dst, file)
        process_code(src_path, dst_path, srcs)

    # extra configs
    dst_path = os.path.join(lib_dst, "extra_operation.py")
    with open(dst_path, "w") as fo:
        code = extra_cutlass_generator.emit_library()
        fo.write(code)
    return dst_prefix


if __name__ == "__main__":
    cutlass_path = os.getenv("SRCDIR")
    output_path = os.getenv("OUT")

    assert output_path is not None
    assert cutlass_path is not None

    mk_cutlass_lib(cutlass_path + "/cutlass", os.path.dirname(output_path))
