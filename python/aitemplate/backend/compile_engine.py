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
from abc import ABC, abstractmethod
from pathlib import Path

from aitemplate.backend.sources import GeneratedSourceFiles
from aitemplate.utils.debug_settings import AITDebugSettings

_DEBUG_SETTINGS = AITDebugSettings()


class CompileEngine(ABC):
    @abstractmethod
    def make_profilers(self, generated_profilers, workdir: Path):
        ...

    @abstractmethod
    def make(
        self,
        generated_sources: GeneratedSourceFiles,
        dll_name: str,
        workdir: Path,
        test_name: str,
        debug_settings: AITDebugSettings = _DEBUG_SETTINGS,
        allow_cache=True,
    ):
        ...
