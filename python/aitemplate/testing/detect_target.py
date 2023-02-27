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
Automatic detect target for testing
"""
import logging
import os
from subprocess import PIPE, Popen

from ..backend.target import CUDA, ROCM

# pylint: disable=W0702, W0612,R1732


_LOGGER = logging.getLogger(__name__)

IS_CUDA = None
FLAG = ""


def _detect_cuda():
    try:
        proc = Popen(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        if "H100" in stdout:
            return "90"
        if any(a in stdout for a in ["A100", "A10G", "RTX 30", "A30"]):
            return "80"
        if "V100" in stdout:
            return "70"
        if "T4" in stdout:
            return "75"
        return None
    except Exception:
        return None


def _detect_rocm():
    try:
        proc = Popen(["rocminfo"], stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        if "gfx90a" in stdout:
            return "gfx90a"
        if "gfx908" in stdout:
            return "gfx908"
        return None
    except Exception:
        return None


def detect_target(**kwargs):
    """Detect GPU target based on nvidia-smi and rocminfo

    Returns
    -------
    Target
        CUDA or ROCM target

    """
    global IS_CUDA, FLAG
    if FLAG:
        if IS_CUDA:
            return CUDA(arch=FLAG, **kwargs)
        else:
            return ROCM(arch=FLAG, **kwargs)
    doc_flag = os.getenv("AIT_BUILD_DOCS", None)
    if doc_flag is not None:
        return CUDA(arch="80", **kwargs)
    flag = _detect_cuda()
    if flag is not None:
        IS_CUDA = True
        FLAG = flag

        _LOGGER.info("Set target to CUDA")
        return CUDA(arch=flag, **kwargs)
    flag = _detect_rocm()
    if flag is not None:
        IS_CUDA = False
        FLAG = flag

        _LOGGER.info("Set target to ROCM")
        return ROCM(arch=flag, **kwargs)
    raise RuntimeError("Unsupported platform")
