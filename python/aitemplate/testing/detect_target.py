# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] Automatic detect target for testing
"""
from subprocess import PIPE, Popen

from ..backend.target import CUDA, ROCM
from ..utils import logger

# pylint: disable=W0702, W0612,R1732


def _detect_cuda():
    """[summary]

    Returns
    -------
    [type]
        [description]
    """
    try:
        proc = Popen(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        if "A100" in stdout or "RTX 30" in stdout:
            return "80"
        if "V100" in stdout:
            return "70"
        if "T4" in stdout:
            return "75"
        return None
    except Exception:
        return None


def _detect_rocm():
    """[summary]

    Returns
    -------
    [type]
        [description]
    """
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
    """[summary]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    RuntimeError
        [description]
    """
    flag = _detect_cuda()
    if flag is not None:
        logger.info(__file__, "Set target to CUDA")
        return CUDA(arch=flag, **kwargs)
    flag = _detect_rocm()
    if flag is not None:
        logger.info(__file__, "Set target to ROCM")
        return ROCM(arch=flag, **kwargs)
    raise RuntimeError("Unsupported platform")
