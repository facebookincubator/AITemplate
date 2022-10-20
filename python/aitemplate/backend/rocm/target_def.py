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
Rocm target specialization.
"""
# pylint: disable=W0702,W0707,W0611,C0415

import os
import re
import shutil
import sys
from typing import List

from aitemplate.backend.target import AIT_STATIC_FILES_PATH

from .. import registry
from ..target import COMPOSABLE_KERNEL_PATH, Target

# pylint: disable=W0613


class ROCM(Target):
    """ROCM target.

    Parameters
    ----------
    Target : Target
        All attributes needed for ROCM.
    """

    def __init__(
        self,
        template_path=COMPOSABLE_KERNEL_PATH,
        arch="GFX908",
        ait_static_files_path=AIT_STATIC_FILES_PATH,
        **kwargs,
    ):
        """Initialize ROCM target.

        Parameters
        ----------
        template_path : str, optional
            Path to composable kernel library, by default "${repo_root}/3rdparty/composable_kernel".
        ait_static_files_path : str
            Absolute path to the AIT static/ directory
        arch : str, optional
            Supported ROCM architecture, by default "GFX908".
        """
        super().__init__(ait_static_files_path)
        self._target_type = 2
        self._template_path = template_path
        self._arch = arch
        self._kwargs = kwargs
        self._compile_options = self._build_compile_options()

    def _pkg_path(self):
        """Initialize package target.

        Returns
        -------
        str
            path to rocm compiler library
        """
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        return rocm_path

    def _build_compile_options(self):
        """Build compilation commands, including compilation flag library and includes.

        Returns
        -------
        List
            List of compilation options.

        Raises
        ------
        RuntimeError
            Unsupported GPU Arch.
        """

        ck_paths = [
            os.path.join(self._template_path),
            os.path.join(self._template_path, "include/"),
            os.path.join(self._template_path, "external/include/half/"),
            os.path.join(self._template_path, "library/include/"),
            os.path.join(self._template_path, "profiler/include/"),
        ]
        options = [
            "-O3",
            "-fPIC",
            "-fvisibility=hidden",
            "-std=c++17",
            "-w",
            "-DCK_TIME_KERNEL=0",
            "-Xclang -mlink-builtin-bitcode -Xclang {}/amdgcn/bitcode/oclc_abi_version_400.bc".format(
                self._pkg_path()
            ),
        ]
        if self._arch in {"GFX908", "gfx908"}:
            options.append("-DCK_AMD_GPU_GFX908")
            options.append("--amdgpu-target=gfx908")
        elif self._arch in {"GFX90a", "gfx90a"}:
            options.append("-DCK_AMD_GPU_GFX90A")
            options.append("--amdgpu-target=gfx90a")
        else:
            raise RuntimeError("Unsupported GPU Arch")
        for path in ck_paths:
            options.append("-I" + path)
        rocrand_path = os.path.join(self._pkg_path(), "rocrand/lib/")
        options.append("-L" + rocrand_path)
        options.append("-lrocrand")
        return " ".join(options)

    def _gen_ck_lib_pkg(self):
        """Build composable kernel python library.

        Raises
        ------
        RuntimeError
            Failed to create ck library.
        """
        self.lib_folder = None
        try:
            import ck_lib  # noqa: F401
        except BaseException:
            try:
                cur_path = os.path.dirname(os.path.realpath(__file__))
                ck_lib_path = os.path.normpath(
                    os.path.join(cur_path, "..", "..", "utils", "mk_ck_lib")
                )
                f_make_lib = registry.get("rocm.make_ck_lib")
                dst_path = f_make_lib(ck_lib_path)
                sys.path.insert(1, dst_path)
            except BaseException as err:
                raise RuntimeError("Failed to create ck library") from err
            self.lib_folder = dst_path

    def __enter__(self):
        """Generate the ck library and generate ck operations."""
        super().__enter__()
        # Generate library.
        self._gen_ck_lib_pkg()
        # Choose the right ops to launch.
        f_gen_ops = registry.get("rocm.gen_ck_ops")
        self._operators = f_gen_ops(self._arch)

    def __exit__(self, ptype, value, trace):
        """Delete the ck library."""
        super().__exit__(ptype, value, trace)
        if self.lib_folder and os.path.exists(self.lib_folder):
            shutil.rmtree(self.lib_folder)

    def cc(self):
        return "hipcc"

    def compile_cmd(self, executable=False):
        """Compile commands.

        Parameters
        ----------
        executable : bool, optional
            Flag of whether to generate executable or obj, by default False.

        Returns
        -------
        str
            Full commands for compilation.
        """
        if executable:
            cmd = self.cc() + " " + self._compile_options + " -o {target} {src}"
        else:
            cmd = self.cc() + " " + self._compile_options + " -c -o {target} {src}"
        return cmd

    def src_extension(self):
        return ".cpp"

    def dev_select_flag(self):
        return "HIP_VISIBLE_DEVICES"

    def select_minimal_algo(self, algo_names: List[str]):
        def comp_func(name):
            compute_args = re.findall(r"_(\d+)_*", name)
            if len(compute_args) != 1:
                raise RuntimeError("Invalid ck op name")
            args = [int(x) for x in compute_args[0]]
            if "Gemm" in name:
                if "GemmPadding" in name:
                    args.insert(0, 0)
                if "GemmDefault" in name:
                    args.insert(0, 1)
            elif "Conv" in name:
                if "ConvFwdDefault" in name:
                    args.insert(0, 0)
                else:
                    args.insert(0, 1)
            else:
                raise RuntimeError("Unknown CK ops.")
            return tuple(args)

        return sorted(algo_names, key=comp_func)[0]


@registry.reg("rocm.create_target")
def create_target(template_path, arch, **kwargs):
    return ROCM(template_path=template_path, arch=arch, **kwargs)
