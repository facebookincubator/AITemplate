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
import sys

from aitemplate import backend, compiler, frontend, testing, utils
from aitemplate._libinfo import __version__  # noqa

if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    PY3STATEMENT = "The minimal Python requirement is Python 3.7"
    raise Exception(PY3STATEMENT)

__all__ = ["backend", "compiler", "frontend", "testing", "utils"]

root_logger = utils.misc.setup_logger(__name__)
