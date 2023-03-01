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
import json
import logging
import os
import pipes
import re
import shutil
import sys
import tempfile

from pathlib import Path
from typing import List

from aitemplate.backend.profiler_cache import ProfileCacheDB

from aitemplate.backend.target import TargetType

from ...utils import environ
from ...utils.misc import is_debug

from .. import registry
from ..target import AIT_STATIC_FILES_PATH, CUTLASS_PATH, Target

# pylint: disable=C0415,W0707,W0611,W0702,W1401


_LOGGER = logging.getLogger(__name__)


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
            os.path.join(self._template_path, "examples/41_fused_multi_head_attention"),
            os.path.join(self._template_path, "examples/45_dual_gemm"),
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
            environ.get_compiler_opt_level(),
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-I" + cutlass_path[0],
            "-I" + cutlass_path[1],
            "-I" + cutlass_path[2],
            "-I" + cutlass_path[3],
            "-I" + cutlass_path[4],
            "-I" + cutlass_path[5],
            "-I" + cutlass_path[6],
        ]
        if self._ndebug == 1:
            options.append("-DNDEBUG")
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
        if self.lib_folder and os.path.exists(self.lib_folder) and not is_debug():
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

        return min(algo_names, key=comp_func)


class FBCUDA(CUDA):
    """FBCUDA target. Used in Meta internal env only."""

    nvcc_option_json = None
    cutlass_path_ = None
    compile_options_ = None

    def __init__(self, arch="80", remote_cache_bytes=None, **kwargs):
        from libfb.py import parutil

        cutlass_src_path = parutil.get_dir_path(
            "aitemplate/AITemplate/fb/3rdparty/cutlass"
        )
        cub_src_path = parutil.get_dir_path("aitemplate/AITemplate/fb/3rdparty/cub")
        static_files_path = parutil.get_dir_path("aitemplate/AITemplate/static")
        self._include_path = None
        if not FBCUDA.cutlass_path_:
            self._include_path = tempfile.mkdtemp()

            FBCUDA.cutlass_path_ = self._include_path + "/cutlass"
            self.cub_path_ = self._include_path + "/cub"
            shutil.copytree(cutlass_src_path, FBCUDA.cutlass_path_)
            shutil.copytree(cub_src_path, self.cub_path_)

            attention_src_path = parutil.get_dir_path(
                "aitemplate/AITemplate/python/aitemplate/backend/cuda/attention/src"
            )
            attention_include_path = self._include_path + "/att_include"
            shutil.copytree(attention_src_path, attention_include_path)
        self.cutlass_path_ = FBCUDA.cutlass_path_

        cutlass_lib_path = parutil.get_dir_path(
            "aitemplate/AITemplate/python/aitemplate/utils/mk_cutlass_lib"
        )
        sys.path.append(cutlass_lib_path)

        if not FBCUDA.nvcc_option_json:
            convert_nvcc_json = parutil.get_file_path(
                os.path.join("aitemplate/testing", "convert_nvcc_cmd")
            )
            _LOGGER.info(f"Load the nvcc compile option from {convert_nvcc_json}")
            with open(convert_nvcc_json, "r") as nvcc_option_json:
                FBCUDA.nvcc_option_json = json.load(nvcc_option_json)
        self.nvcc_options_json = FBCUDA.nvcc_option_json

        self.remote_cache_bytes = remote_cache_bytes
        super().__init__(self.cutlass_path_, static_files_path, arch, **kwargs)

    def _build_compile_options(self):
        if not FBCUDA.compile_options_:
            cutlass_path = [
                os.path.join(self._template_path, "include"),
                os.path.join(self._template_path, "tools/util/include"),
                os.path.join(self._template_path, "examples/35_gemm_softmax"),
                os.path.join(
                    self._template_path, "examples/41_fused_multi_head_attention"
                ),
                os.path.join(self._template_path, "examples/45_dual_gemm"),
                os.path.join(self._template_path, "../att_include"),
                os.path.join(self._template_path, "../att_include/fmha"),
                os.path.join(self._template_path, "../cub"),
            ]
            fb_include_path = os.path.join(self._include_path, "fb_include")
            pp_args = self.nvcc_options_json["pp_args"]
            with open(fb_include_path, "w") as fb_include:
                for arg in pp_args:
                    fb_include.write(pipes.quote(arg) + "\n")

            options = self.nvcc_options_json["args"] + [
                "-I" + cutlass_path[0],
                "-I" + cutlass_path[1],
                "-I" + cutlass_path[2],
                "-I" + cutlass_path[3],
                "-I" + cutlass_path[4],
                "-I" + cutlass_path[5],
                "-I" + cutlass_path[6],
                f"-Xcompiler '-Wp\,@{fb_include_path}'",  # noqa: W605
                "-Xcompiler -Wno-strict-aliasing",
                "-Xcompiler -Wno-narrowing",
                "-Xcompiler -Wno-error=maybe-uninitialized",
                "-Xcompiler -Wno-uninitialized",
                "-Xcompiler -Wno-error=array-bounds",
                "-Xcompiler -fPIC",
                "-Xcompiler -fvisibility=hidden",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",
                "-w",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                "-gencode=arch=compute_%s,code=[sm_%s,compute_%s]"
                % (self._arch, self._arch, self._arch),
                "-Xcompiler=-Wconversion",
                environ.get_compiler_opt_level(),
                "-std=c++17",
            ]
            if self._ndebug == 1:
                options.append("-DNDEBUG")
            FBCUDA.compile_options_ = " ".join(options)
        compile_options = FBCUDA.compile_options_
        _LOGGER.info(f"The compile options are: {compile_options}")
        return compile_options

    def __exit__(self, ptype, value, trace):
        super().__exit__(ptype, value, trace)

    def binary_compile_cmd(self):
        """
        There is no ld by default in the prod env. Instead, we use ld from the gvfs path.
        """
        ld = self.nvcc_options_json["ld"]
        return " ".join([ld, "-r -b binary -o {target} {src}"])

    def cc(self):
        return self.nvcc_options_json["nvcc_bin"]

    def make(self):
        return self.nvcc_options_json.get("make_bin", super().make())

    def compile_options(self):
        return self._compile_options

    def get_custom_libs(self, absolute_dir, filename) -> str:
        def list_rindex(input_list, x):
            for i in reversed(range(len(input_list))):
                if input_list[i] == x:
                    return i
            raise ValueError("{} is not in list".format(x))

        from libfb.py import parutil

        absolute_dir = os.path.normpath(absolute_dir)
        dir_parts = Path(absolute_dir).parts
        relative_path = Path(
            "/".join(dir_parts[list_rindex(dir_parts, "aitemplate") :]) + "/" + filename
        )
        f_name = parutil.get_dir_path(relative_path)
        with open(f_name) as f:
            res = f.read()
            return res

    def in_ci_env(self):
        return (
            os.environ.get("INSIDE_RE_WORKER", None) == "1" and not self.trick_ci_env()
        )

    @classmethod
    def remote_logger(cls, record):
        """
        Upload the record to Scuba table
        """
        # Only upload when force_profile or trick_ci_env is specified.
        # i.e. FORCE_PROFILE=1 or -c aitemplate.force_profile=true or TRICK_CI_ENV=1
        # Otherwise, dummy profiling records are not useful.
        if cls.force_profile(cls) or cls.trick_ci_env(cls):
            from aitemplate.AITemplate.fb.remote_logger import AITemplateRemoteLogger

            try:
                AITemplateRemoteLogger.log(record)
            except Exception as e:
                _LOGGER.info(f"remote_logger failed: {e}")

    def _load_profile_cache(self):
        """Load local profile cache for this target."""
        cache_path = self._prepare_profile_cache_path()
        if cache_path is None:
            return

        if self.remote_cache_bytes is not None:
            _LOGGER.info(
                f"Loading profile cache from provided cache content with length {len(self.remote_cache_bytes)}",
            )
            with open(cache_path, "wb") as f:
                f.write(self.remote_cache_bytes)
        _LOGGER.info(f"Loading profile cache from: {cache_path}")
        self._profile_cache = ProfileCacheDB(
            TargetType(self._target_type).name, path=cache_path
        )


@registry.reg("fb.cuda.create_target")
def create_target_fb(arch, **kwargs):
    return FBCUDA(arch=arch, **kwargs)


@registry.reg("cuda.create_target")
def create_target(template_path, arch, **kwargs):
    return CUDA(template_path=template_path, arch=arch, **kwargs)
