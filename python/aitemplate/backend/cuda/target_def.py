# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] cuda target specialization
"""
import json
import os
import pipes
import re
import shutil
import sys
import tempfile

from pathlib import Path
from typing import List

from ...utils import logger

from .. import registry
from ..target import AIT_STATIC_FILES_PATH, CUTLASS_PATH, Target

# pylint: disable=C0415,W0707,W0611,W0702,W1401


class CUDA(Target):
    """[summary]

    Parameters
    ----------
    Target : [type]
        [description]
    """

    def __init__(
        self,
        template_path=CUTLASS_PATH,
        ait_static_files_path=AIT_STATIC_FILES_PATH,
        arch="80",
        **kwargs,
    ):
        """[summary]

        Parameters
        ----------
        template_path : str, optional
            [description], by default "${repo_root}/3rdparty/cutlass"
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
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
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
        """[summary]

        Parameters
        ----------
        executable : bool, optional
            [description], by default False
        """
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


class FBCUDA(CUDA):
    """[summary]

    Parameters
    ----------
    Target : [type]
        [description]
    """

    def __init__(self, arch="80", **kwargs):
        from libfb.py import parutil

        cutlass_src_path = parutil.get_dir_path(
            "aitemplate/AITemplate/fb/3rdparty/cutlass"
        )
        cub_src_path = parutil.get_dir_path("aitemplate/AITemplate/fb/3rdparty/cub")
        static_files_path = parutil.get_dir_path("aitemplate/AITemplate/static")

        self._include_path = tempfile.mkdtemp()

        self.cutlass_path_ = self._include_path + "/cutlass"
        self.cub_path_ = self._include_path + "/cub"
        shutil.copytree(cutlass_src_path, self.cutlass_path_)
        shutil.copytree(cub_src_path, self.cub_path_)

        attention_src_path = parutil.get_dir_path(
            "aitemplate/AITemplate/python/aitemplate/backend/cuda/attention/src"
        )
        attention_include_path = self._include_path + "/att_include"
        shutil.copytree(attention_src_path, attention_include_path)

        cutlass_lib_path = parutil.get_dir_path(
            "aitemplate/AITemplate/python/aitemplate/utils/mk_cutlass_lib"
        )
        sys.path.append(cutlass_lib_path)

        convert_nvcc_json = parutil.get_file_path(
            os.path.join("aitemplate/testing", "convert_nvcc_cmd")
        )
        logger.info(__name__, f"Load the nvcc compile option from {convert_nvcc_json}")
        with open(convert_nvcc_json, "r") as nvcc_option_json:
            self.nvcc_options_json = json.load(nvcc_option_json)

        super().__init__(self.cutlass_path_, static_files_path, arch, **kwargs)

    def _build_compile_options(self):
        cutlass_path = [
            os.path.join(self._template_path, "include"),
            os.path.join(self._template_path, "tools/util/include"),
            os.path.join(self._template_path, "examples/35_gemm_softmax"),
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
            "-O3",
            "-std=c++17",
        ]
        logger.debug(__name__, "The comiple options are: {" ".join(options)}")
        return " ".join(options)

    def __exit__(self, ptype, value, trace):
        super().__exit__(ptype, value, trace)
        shutil.rmtree(self._include_path)

    def binary_compile_cmd(self):
        """
        There is no ld by default in the prod env. Instead, we use ld from the gvfs path.
        """
        ld = self.nvcc_options_json["ld"]
        return " ".join([ld, "-r -b binary -o {target} {src}"])

    def cc(self):
        return self.nvcc_options_json["nvcc_bin"]

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
        return os.environ.get("INSIDE_RE_WORKER", None) == "1"

    @classmethod
    def remote_logger(cls, record):
        """
        Upload the record to Scuba table
        """
        # Only upload when force_profile is specified.
        # i.e. FORCE_PROFILE=1 or -c aitemplate.force_profile=true
        # Otherwise, dummy profiling records are not useful.
        if cls.force_profile(cls):
            from aitemplate.AITemplate.fb.remote_logger import AITemplateRemoteLogger

            try:
                AITemplateRemoteLogger.log(record)
            except Exception as e:
                logger.info(__name__, f"remote_logger failed: {e}")


@registry.reg("fb.cuda.create_target")
def create_target_fb(arch, **kwargs):
    return FBCUDA(arch=arch, **kwargs)


@registry.reg("cuda.create_target")
def create_target(template_path, arch, **kwargs):
    return CUDA(template_path=template_path, arch=arch, **kwargs)
