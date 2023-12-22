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

from aitemplate.backend import registry

from aitemplate.backend.profiler_cache import ProfileCacheDB

from aitemplate.backend.target import (
    AIT_STATIC_FILES_PATH,
    CUTLASS_PATH,
    Target,
    TargetType,
)

from aitemplate.utils import environ
from aitemplate.utils.misc import is_debug, is_linux

# pylint: disable=C0415,W0707,W0611,W0702,W1401


_LOGGER = logging.getLogger(__name__)


class CUDA(Target):
    """CUDA target."""

    def __init__(
        self,
        template_path=CUTLASS_PATH,
        ait_static_files_path=AIT_STATIC_FILES_PATH,
        arch="80",
        cuda_version=None,
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
        if cuda_version is None:
            # try to set default CUDA version based on the arch
            if arch == "80":
                cuda_version = "11.4.2"
            elif arch == "90":
                cuda_version = "12.0.0"
        self._cuda_version = cuda_version

    def _build_include_directories(self) -> List[str]:
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
        ]
        ait_static_path = os.path.join(self._ait_include_path, "include/kernels")

        output = [ait_static_path]
        output.extend(cutlass_path)
        return output

    def get_include_directories(self) -> List[str]:
        return self._build_include_directories()

    def _build_gnu_host_compiler_options(self) -> List[str]:
        return [
            "-fPIC",
            "-Wconversion",
            "-fno-strict-aliasing",
            "-fvisibility=hidden",
        ]

    def get_host_compiler_options(self) -> List[str]:
        return self._build_gnu_host_compiler_options()

    def _get_nvcc_debug_options(self) -> str:
        CUDA_DEBUG_LEVEL_STRINGS = ["", "-lineinfo", "-g -G"]
        level = environ.get_cuda_nvcc_debug_level()
        if level.isdigit():
            level = int(level)
            assert (
                level >= 0 and level < 3
            ), "Debug level out of range. Must be 0 (no debug info), 1 (lineinfo) or 2 (with debug info, disable opt)"
            return CUDA_DEBUG_LEVEL_STRINGS[level]
        return level

    def _build_nvcc_compiler_options(self) -> List[str]:
        code = [f"sm_{self._arch}", f"compute_{self._arch}"]
        if environ.enable_cuda_lto():
            code += [f"lto_{self._arch}"]
        options = [
            "-t=0",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-w",
            f"-gencode=arch=compute_{self._arch},code=[{','.join(code)}]",
            environ.get_compiler_opt_level(),
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ]
        if environ.enable_ptxas_info():
            options.extend(
                [
                    "--keep",  # Keep the intermediate files for debugging (including ptx, sass, cubin etc.)
                    "--ptxas-options=--warn-on-local-memory-usage",  # warn us if local memory is used in CUDA Kernels
                    "--ptxas-options=--warn-on-spills",  # warn us if register spilling happens in CUDA Kernels
                    "--resource-usage",  # Report on CUDA resource usage (shared mem, registers etc.)
                    "--source-in-ptx",
                ]
            ),  # Annotate the ptx file with source information
        options.append(self._get_nvcc_debug_options())
        if self._ndebug == 1:
            options.append("-DNDEBUG")
        if environ.use_fast_math() and (
            "use_fast_math" not in self._kwargs or self._kwargs["use_fast_math"]
        ):
            options.extend(
                [
                    "--use_fast_math",
                    "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",
                    "-DAIT_USE_FAST_MATH=1",
                ]
            )
        return options

    def get_device_compiler_options(self) -> List[str]:
        return self._build_nvcc_compiler_options()

    def _build_compile_options(self):
        include_paths = self._build_include_directories()
        host_compiler_options = self._build_gnu_host_compiler_options()
        nvcc_compiler_options = self._build_nvcc_compiler_options()

        options = (
            nvcc_compiler_options
            + [
                f"-Xcompiler {opt}" if "=" in opt else f"-Xcompiler={opt}"
                for opt in host_compiler_options
            ]
            + ["-I" + path for path in include_paths]
        )

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
        self._operators = f_gen_ops(self._arch, self._cuda_version)

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
    static_compile_options_ = None
    optimize_for_compilation_time_ = False

    def __init__(self, arch="80", remote_cache_bytes=None, **kwargs):
        from libfb.py import parutil

        cutlass_src_path = parutil.get_dir_path(
            "aitemplate/AITemplate/fb/3rdparty/cutlass"
        )
        cub_src_path = parutil.get_dir_path("aitemplate/AITemplate/fb/3rdparty/cub")
        static_files_path = parutil.get_dir_path("aitemplate/AITemplate/static")
        if "optimize_for_compilation_time" in kwargs:
            FBCUDA.optimize_for_compilation_time_ = kwargs[
                "optimize_for_compilation_time"
            ]
        _LOGGER.info(
            "Optimize for compilation time : {}".format(
                FBCUDA.optimize_for_compilation_time_
            )
        )
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
            ait_static_include_path = self._include_path + "/static"
            shutil.copytree(
                static_files_path + "/include/kernels", ait_static_include_path
            )
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
        cuda_version = self.nvcc_option_json.get("cuda_version", None)

        self.remote_cache_bytes = remote_cache_bytes
        super().__init__(
            template_path=self.cutlass_path_,
            ait_static_files_path=static_files_path,
            arch=arch,
            cuda_version=cuda_version,
            **kwargs,
        )

    def _build_include_directories_from_sourcetree(self) -> List[str]:
        my_path: Path = Path(os.path.realpath(__file__))  # noqa
        ait_basepath: Path = my_path.parent.parent.parent.parent.parent.absolute()
        assert (
            ait_basepath.name == "AITemplate"
        ), "AITemplate basepath resolution failed"
        relative_include_paths = [
            "fb/3rdparty/cutlass/examples/35_gemm_softmax",
            "fb/3rdparty/cutlass/examples/41_fused_multi_head_attention",
            "fb/3rdparty/cutlass/examples/45_dual_gemm",
            "fb/3rdparty/cutlass/examples/common",
            "fb/3rdparty/cutlass/include",
            "fb/3rdparty/cutlass/tools/library/include",
            "fb/3rdparty/cutlass/tools/library/src",
            "fb/3rdparty/cutlass/tools/util/include",
            "python/aitemplate/backend/cuda/attention/src",
            "python/aitemplate/backend/cuda/attention/src/fmha",
            "static/include",
            "static/include/kernels",
        ]
        include_paths = [
            str((ait_basepath / ipath).absolute()) for ipath in relative_include_paths
        ]
        return include_paths

    def _build_include_directories(self) -> List[str]:
        if environ.enable_include_from_sourcetree():
            return self._build_include_directories_from_sourcetree()
        cutlass_path = [
            os.path.join(self._template_path, "include"),
            os.path.join(self._template_path, "tools/util/include"),
            os.path.join(self._template_path, "examples/35_gemm_softmax"),
            os.path.join(self._template_path, "examples/41_fused_multi_head_attention"),
            os.path.join(self._template_path, "examples/45_dual_gemm"),
            os.path.join(self._template_path, "../att_include"),
            os.path.join(self._template_path, "../att_include/fmha"),
        ]
        if self._include_path is not None:
            ait_static_path = os.path.join(self._include_path, "static")
            return [ait_static_path] + cutlass_path
        else:
            return cutlass_path

    def get_include_directories(self) -> List[str]:
        return self._build_include_directories()

    def get_host_compiler_options(self) -> List[str]:
        # a placeholder
        raise NotImplementedError

    def get_device_compiler_options(self) -> List[str]:
        # a placeholder
        raise NotImplementedError

    def _build_compile_options(self):
        if not FBCUDA.static_compile_options_:
            include_paths = self._build_include_directories()
            fb_include_path = os.path.join(self._include_path, "fb_include")
            pp_args = self.nvcc_options_json["pp_args"]
            with open(fb_include_path, "w") as fb_include:
                for arg in pp_args:
                    fb_include.write(pipes.quote(arg) + "\n")

            nvcc_arch = self._arch
            if nvcc_arch == "90":
                # required by CUTLASS SM90 TMA kernels
                nvcc_arch = "90a"

            options = (
                self.nvcc_options_json["args"]
                + ["-I" + path for path in include_paths]
                + [
                    f"-Xcompiler '-Wp\\,@{fb_include_path}'",
                    "-Xcompiler -Wno-strict-aliasing",
                    "-Xcompiler -Wno-narrowing",
                    "-Xcompiler -Wno-error=maybe-uninitialized",
                    "-Xcompiler -Wno-uninitialized",
                    "-Xcompiler -Wno-error=array-bounds",
                    "-Xcompiler -fPIC",
                    "-Xcompiler -fvisibility=hidden",
                    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                    "-w",
                    "--expt-relaxed-constexpr",
                    f"-gencode=arch=compute_{nvcc_arch},code=[sm_{nvcc_arch},compute_{nvcc_arch}]",
                    "-Xcompiler=-Wconversion",
                    environ.get_compiler_opt_level()
                    if not FBCUDA.optimize_for_compilation_time_
                    else "-O1",
                    "-std=c++17",
                ]
                + (
                    ["-DOPTIMIZE_FOR_COMPILATION_TIME"]
                    if FBCUDA.optimize_for_compilation_time_
                    else []
                )
            )
            if environ.enable_ptxas_info():
                options.extend(
                    [
                        "--keep",  # Keep the intermediate files for debugging (including ptx, sass, cubin etc.)
                        "--ptxas-options=--warn-on-local-memory-usage",  # warn us if local memory is used in CUDA Kernels
                        "--ptxas-options=--warn-on-spills",  # warn us if register spilling happens in CUDA Kernels
                        "--resource-usage",  # Report on CUDA resource usage (shared mem, registers etc.)
                        "--source-in-ptx",  # Annotate the ptx file with source information
                    ]
                ),
            options.append(self._get_nvcc_debug_options())
            if self._ndebug == 1:
                options.append("-DNDEBUG")
            FBCUDA.static_compile_options_ = options
        compile_options = list(FBCUDA.static_compile_options_)
        if environ.use_fast_math() and (
            "use_fast_math" not in self._kwargs or self._kwargs["use_fast_math"]
        ):
            compile_options.extend(
                [
                    "--use_fast_math",
                    "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",
                    "-DAIT_USE_FAST_MATH=1",
                ]
            )
        compile_options_str = " ".join(compile_options)
        _LOGGER.info(f"The compile options are: {compile_options_str}")
        return compile_options_str

    def __exit__(self, ptype, value, trace):
        super().__exit__(ptype, value, trace)

    def binary_compile_cmd(self):
        """
        There is no ld by default in the prod env. Instead, we use ld from the gvfs path.
        """
        ld = self.nvcc_options_json["ld"]
        objcopy = self.nvcc_options_json["objcopy"]
        cmd = " ".join([ld, "-r -b binary -o {target} {src}"])
        # Support models with >2GB constants on Linux only
        if is_linux():
            cmd += (
                f" && {objcopy} --rename-section"
                " .data=.lrodata,alloc,load,readonly,data,contents"
                " {target} {target}"
            )
        return cmd

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

    def in_ci_env(self) -> bool:
        return (
            os.environ.get("INSIDE_RE_WORKER", None) == "1" and not self.trick_ci_env()
        )

    def postprocess_build_dir(self, build_dir: str) -> None:
        # Write a standard TARGETS file to enable standalone exe code navigation
        from aitemplate.backend import buck_support

        additional_build_dir_contents = {"TARGETS": buck_support.AIT_BUILD_DIR_TARGETS}
        for filename, content in additional_build_dir_contents.items():
            filepath = os.path.join(build_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        if environ.enable_cuda_source_navigation_fix():
            # We rename all .cu files to cu.h, and write a .cu
            # file in their stead that only includes this cu.h file.
            # The purpose is to enable .cu source navigation for certain IDEs..
            build_dir_path = Path(build_dir)
            cu_files = list(build_dir_path.glob("*.cu"))
            for p in cu_files:
                corresponding_include_file = p.with_name(p.name + ".h")
                if corresponding_include_file.exists():
                    corresponding_include_file.unlink()
                # rename .cu file to .cu.h
                p.rename(corresponding_include_file)
                # write .cu file which just includes the original, now found
                # under .cu.h
                p.write_text(f'#include "{corresponding_include_file.name}"\n')

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
