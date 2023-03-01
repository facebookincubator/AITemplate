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
