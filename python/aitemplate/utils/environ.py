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
