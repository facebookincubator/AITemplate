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

import importlib.machinery
import logging
import os

import torch

logger: logging.Logger = logging.getLogger(__name__)


def is_oss_ait_model():
    return False


def _get_extension_path(lib_name):

    lib_dir = os.path.dirname(__file__)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError

    return ext_specs.origin


if torch.version.hip is None:
    # For Meta internal workloads, we don't have an active plan to apply AITemplate on AMD GPUs.
    # As such, for AMD build we skip all AITemplate related supports. T186819748 is used to
    # track the plans/strategies for AITemplate enablement on AMD GPUs if needed in the future.
    try:
        torch.ops.load_library("//deeplearning/ait:AITModel")
        logger.info("===Load non-OSS AITModel===")

    except (ImportError, OSError):
        lib_path = _get_extension_path("libait_model")
        torch.ops.load_library(lib_path)
        logger.info("===Load OSS AITModel===")

        def is_oss_ait_model():  # noqa: F811
            return True
