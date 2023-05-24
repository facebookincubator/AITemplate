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
A common place for holding AIT-related env control variables
"""
import logging
import os
from typing import Optional


_LOGGER = logging.getLogger(__name__)


def get_compiler_opt_level() -> str:
    # The reason: it is typical in our situation that an option
    # --optimize <level> (-Ox) is for a HOST compiler. And -O3 does
    # literally nothing except for the enormous compilation time.
    #
    # So, it is safe to allow users to override this option in order
    # to significantly speedup the computations / testing, especially
    # for very large models.
    compiler_opt = os.getenv("AIT_COMPILER_OPT", "-O3")

    return compiler_opt


def use_fast_math() -> str:
    """
    Whether the fast math option should be used for the device code generation.
    Fast math implies the use of approximate math operations (say,
    a division operation), allowing to gain speed at the cost of accuracy.
    Default value is "1".
    """
    return os.getenv("AIT_USE_FAST_MATH", "1") == "1"


def enable_cuda_lto() -> bool:
    """
    nvcc will use LTO flags during compilation
    Default value is "0".
    """
    return os.getenv("AIT_ENABLE_CUDA_LTO", "0") == "1"


def force_profiler_cache() -> bool:
    """
    Force the profiler to use the cached results. The profiler will throw
    a runtime exception if it cannot find cached results. This env may be
    useful to capture any cache misses due to cache version updates or
    other relevant code changes.
    """
    force_cache = os.environ.get("AIT_FORCE_PROFILER_CACHE", None) == "1"
    if force_cache:
        assert (
            os.environ.get("FORCE_PROFILE", None) != "1"
        ), "cannot specify both AIT_FORCE_PROFILER_CACHE and FORCE_PROFILE"
    _LOGGER.info(f"{force_cache=}")
    return force_cache


def time_compilation() -> bool:
    """
    When enabled, time each make command at compilation time.
    This helps us doing compilation time analysis.
    Requires to install "time".
    """
    return os.getenv("AIT_TIME_COMPILATION", "0") == "1"


def shorten_tensor_names_for_plots() -> bool:
    """
    When enabled, long tensor names will be replaced with a hash string,
    making the graph representation significantly simpler.
    """
    return os.getenv("AIT_PLOT_SHORTEN_TENSOR_NAMES", "0") == "1"


def ait_build_cache_dir() -> Optional[str]:
    """
    When set to a non-empty string, cache the build artifacts
    below this directory for significantly faster builds.

    See aitemplate.backend.build_cache

    Returns:
        Optional[str]: Value of AIT_BUILD_CACHE_DIR environment variable,
        or None if not set.
    """
    return os.environ.get("AIT_BUILD_CACHE_DIR", None)


def ait_build_cache_skip_percentage() -> int:
    """
    When set to a non-empty string, and if AIT_BUILD_CACHE_DIR
    is set, the build cache will be skipped randomly with
    a probability correspinding to the specified percentage

    Returns:
        int: Integer value of AIT_BUILD_CACHE_SKIP_PERCENTAGE environment variable,
        or 5 if not set.
    """
    return int(os.environ.get("AIT_BUILD_CACHE_SKIP_PERCENTAGE", "30"))


def ait_build_cache_skip_profiler() -> bool:
    """
    boolean value of AIT_BUILD_CACHE_SKIP_PROFILER environment variable.
    Will return True if that variable is not set, if it is equal to "0",
    an empty string or "False" ( case insensitive ). Will return True
    in all other cases.
    """
    ret = os.environ.get("AIT_BUILD_CACHE_SKIP_PROFILER", "1")
    if ret is None or ret == "" or ret == "0" or ret.lower() == "false":
        return False
    return True


def ait_build_cache_max_mb() -> int:
    """
    boolean value of AIT_BUILD_CACHE_MAX_MB environment variable.
    This determines the maximum size of the artifact data to be cached
    in MB. For larger (raw, uncompressed) data the build cache will
    be skipped. Defaults to 30.
    """
    return int(os.environ.get("AIT_BUILD_CACHE_MAX_MB", "30"))


def allow_cutlass_sm90_kernels() -> bool:
    """
    Whether the SM90 CUTLASS kernels should to be considered
    alongside the SM80 CUTLASS kernels on the CUDA arch 90
    (for the CUDA back-end of the GEMM ops). Default: False.
    """
    return (
        force_cutlass_sm90_kernels()
        or os.getenv("AIT_ALLOW_CUTLASS_SM90_KERNELS", "0") == "1"
    )


def force_cutlass_sm90_kernels() -> bool:
    """
    Whether only the SM90 CUTLASS kernels (and not the SM80 ones)
    should be considered on the CUDA arch 90 (for the CUDA
    back-end of the GEMM ops). Default: False.
    """
    return os.getenv("AIT_FORCE_CUTLASS_SM90_KERNELS", "0") == "1"


def multistream_mode() -> int:
    """
    Multi-stream mode. 0 - no multistream. 1 - simple multistream.
    Default: 0.
    """

    # temporarily override it in order to test
    return int(os.getenv("AIT_MULTISTREAM_MODE", "0"))


def multistream_additional_streams() -> int:
    """
    Number of extra streams in multi-stream mode.

    This option is independent from AIT_MULTISTREAM_MAX_MEM_PARALLEL_OPS.

    For example, say, there are 100 ops that can be run in parallel.

    Example 1: AIT_MULTISTREAM_EXTRA_STREAMS=4 and AIT_MULTISTREAM_MAX_MEM_PARALLEL_OPS=100.
    In this case 5 streams will be used (1 base and 4 extra),
    every stream gets 20 operators and no inter-stream barriers are used.
    Memory planning is done for 100 parallel ops.

    Example 2: AIT_MULTISTREAM_EXTRA_STREAMS=4 and AIT_MULTISTREAM_MAX_MEM_PARALLEL_OPS=5.
    In this case 5 streams will be used (1 base and 4 extra),
    there will be 20 waves separated by inter-stream barriers,
    every stream gets 1 operator for every wave.
    Memory planning is done for 20 waves of 5 parallel ops each.

    """
    return int(os.getenv("AIT_MULTISTREAM_EXTRA_STREAMS", "4"))


def multistream_max_mem_parallel_ops() -> int:
    """
    Maximum number of parallel operators used in memory planning
    for simple multi-stream mode.
    Larger value imply higher level of possible parallelism, but
    higher memory allocations.

    This option is independent from AIT_MULTISTREAM_EXTRA_STREAMS.

    For example, say, there are 100 ops that can be run in parallel.

    Example 1: AIT_MULTISTREAM_EXTRA_STREAMS=4 and AIT_MULTISTREAM_MAX_MEM_PARALLEL_OPS=100.
    In this case 5 streams will be used (1 base and 4 extra),
    every stream gets 20 operators and no inter-stream barriers are used.
    Memory planning is done for 100 parallel ops.

    Example 2: AIT_MULTISTREAM_EXTRA_STREAMS=4 and AIT_MULTISTREAM_MAX_MEM_PARALLEL_OPS=5.
    In this case 5 streams will be used (1 base and 4 extra),
    there will be 20 waves separated by inter-stream barriers,
    every stream gets 1 operator for every wave.
    Memory planning is done for 20 waves of 5 parallel ops each.
    """
    # unlimited by default
    return int(os.getenv("AIT_MULTISTREAM_MAX_MEM_PARALLEL_OPS", "99999999"))


def is_cmake_compilation() -> bool:
    """
    When enabled, compiles the model via invoking CMake rather than
    invoking make directly.
    """

    # todo: replace with more builders?
    return os.getenv("AIT_USE_CMAKE_COMPILATION", "0") == "1"
