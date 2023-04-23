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

from __future__ import annotations

from typing import List, Set, Union


class GeneratedSourceFiles:
    def __init__(self):
        self._sources: Set[str] = set()
        self._headers: Set[str] = set()
        self._others: Set[str] = set()

    def _combine(self, other: GeneratedSourceFiles) -> None:
        for source_fn in other._sources:
            self._sources.add(source_fn)
        for header_fn in other._headers:
            self._headers.add(header_fn)
        for other_fn in other._others:
            self._others.add(other_fn)

    def _add_one(self, filename: str) -> None:
        if filename.endswith(".h"):
            self._headers.add(filename)
        elif (
            filename.endswith(".cu")
            or filename.endswith(".c")
            or filename.endswith(".cpp")
        ):
            self._sources.add(filename)
        else:
            self._others.add(filename)

    def add(self, items: Union[str, List[str], GeneratedSourceFiles]) -> None:
        if isinstance(items, GeneratedSourceFiles):
            self._combine(items)
        elif isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    self._add_one(item)
                else:
                    raise TypeError()
        elif isinstance(items, str):
            self._add_one(items)
        else:
            raise TypeError()

    @property
    def headers(self) -> Set[str]:
        return self._headers

    @property
    def sources(self) -> Set[str]:
        return self._sources

    @property
    def others(self) -> Set[str]:
        return self._others
