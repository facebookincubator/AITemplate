# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""[summary]
"""
import logging
import os
import sys

from . import backend, compiler, frontend, frontend_v2, testing, utils
from ._libinfo import __version__  # noqa

if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    PY3STATEMENT = "The minimal Python requirement is Python 3.7"
    raise Exception(PY3STATEMENT)

__all__ = ["backend", "compiler", "frontend", "testing", "utils", "frontend_v2"]

root_logger = logging.getLogger(__name__)
info_handle = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s <%(name)s> %(message)s")
info_handle.setFormatter(formatter)
root_logger.addHandler(info_handle)
root_logger.propagate = False

DEFAULT_LOGLEVEL = logging.getLogger().level
log_level_str = os.environ.get("LOGLEVEL", None)
LOG_LEVEL = (
    getattr(logging, log_level_str.upper())
    if log_level_str is not None
    else DEFAULT_LOGLEVEL
)
root_logger.setLevel(LOG_LEVEL)
