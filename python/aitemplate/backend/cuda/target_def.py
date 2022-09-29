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
CUDA target specialization
"""

import os
import re
import shutil
import sys

from typing import List

from .. import registry
from ..target import AIT_STATIC_FILES_PATH, CUTLASS_PATH, Target

# pylint: disable=C0415,W0707,W0611,W0702,W1401


class CUDA(Target):
    """CUDA target."""

    def __init__(
        self,
        template_path=CUTLASS_PATH,
        ait_static_files_path=AIT_STATIC_FILES_PATH,
        arch="80",
        **kwargs,
    ):
        """CUDA target init.

        Parameters
        ----------
        template_path : str, optional
            by default "${repo_root}/3rdparty/cutlass"
        ait_static_files_path : str
            Absolute path to the AIT static/ directory
        """
        super().__init__(ait_static_files_path)
        self._target_type = 1
        self._template_path = template_path
        self._ait_include_path = ait_static_files_path
        self._arch = arch
        self._kwargs = kwargs
        self._compile_options = self._build_compile_options()

    def _build_compile_options(self):
        flash_attention_path = ""
        if os.path.exists(
            os.path.join(
                self._template_path,
                "../../python/aitemplate/backend/cuda/attention/src",
            )
        ):
            # setup develop
            flash_attention_path = os.path.join(
                self._template_path,
                "../../python/aitemplate/backend/cuda/attention/src",
            )
        else:
            # in wheel
            flash_attention_path = os.path.join(
                self._template_path, "../../backend/cuda/attention/src"
            )
        cutlass_path = [
            os.path.join(self._template_path, "include"),
            os.path.join(self._template_path, "tools/util/include"),
            os.path.join(self._template_path, "examples/35_gemm_softmax"),
            os.path.join(
                flash_attention_path,
                "./",
            ),
            os.path.join(
                flash_attention_path,
                "fmha",
            ),
            os.path.join(self._template_path, "../cub"),
        ]
        options = [
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",
            "-w",
            "-gencode=arch=compute_%s,code=[sm_%s,compute_%s]"
            % (self._arch, self._arch, self._arch),
            "-Xcompiler=-fPIC",
            "-Xcompiler=-Wconversion",
            "-Xcompiler=-fno-strict-aliasing",
            "-Xcompiler -fvisibility=hidden",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-I" + cutlass_path[0],
            "-I" + cutlass_path[1],
            "-I" + cutlass_path[2],
            "-I" + cutlass_path[3],
            "-I" + cutlass_path[4],
        ]
        return " ".join(options)

    def src_extension(self):
        return ".cu"

    def _gen_cutlass_lib_pkg(self):
        self.lib_folder = None
        try:
            import cutlass_lib  # noqa: F401
        except Exception:
            try:
                f_make_lib = registry.get("cuda.make_cutlass_lib")
                dst_path = f_make_lib(self.template_path())
                sys.path.insert(1, dst_path)
            except Exception as err:
                raise RuntimeError(
                    "Failed to create cutlass library lib: {}".format(err)
                ) from err
            self.lib_folder = dst_path

    def __enter__(self):
        super().__enter__()
        self._gen_cutlass_lib_pkg()
        f_gen_ops = registry.get("cuda.gen_cutlass_ops")
        self._operators = f_gen_ops(self._arch)

    def __exit__(self, ptype, value, trace):
        super().__exit__(ptype, value, trace)
        if self.lib_folder and os.path.exists(self.lib_folder):
            shutil.rmtree(self.lib_folder)

    def cc(self):
        return "nvcc"

    def compile_cmd(self, executable=False):
        if executable:
            cmd = self.cc() + " " + self._compile_options + " -o {target} {src}"
        else:
            cmd = self.cc() + " " + self._compile_options + " -c -o {target} {src}"
        return cmd

    def dev_select_flag(self):
        return "CUDA_VISIBLE_DEVICES"

    def select_minimal_algo(self, algo_names: List[str]):
        def comp_func(name):
            compute_args = re.findall(r"(\d+)x(\d+)_(\d+)x(\d+)", name)
            if len(compute_args) != 1:
                raise RuntimeError("Invalid cutlass op name")
            args = [int(x) for x in compute_args[0]]
            align_args = name.split("_")
            args.append(int(align_args[-2]))
            args.append(int(align_args[-1]))
            return tuple(args)

        return sorted(algo_names, key=comp_func)[0]


@registry.reg("cuda.create_target")
def create_target(template_path, arch, **kwargs):
    return CUDA(template_path=template_path, arch=arch, **kwargs)
