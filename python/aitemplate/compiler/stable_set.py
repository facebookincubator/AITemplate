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
A stable set is like a Python set which produces deterministic results.
It also tries to preserve the original element order as much as possible, which could
potentially make debugging (e.g. comparison with the original graph, comparison between
AIT GPU trace and other GPU traces) easier.
"""
from collections import abc
from typing import Any, Iterable


class StableSet(abc.MutableSet):
    def __init__(self, s: Iterable[Any] = None):
        if s is None:
            s = []
        self._d = {item: None for item in s}

    def add(self, value) -> None:
        self._d[value] = None

    def update(self, other) -> None:
        for item in other:
            self._d[item] = None

    def discard(self, value) -> None:
        self._d.pop(value, None)

    def remove(self, value) -> None:
        self._d.pop(value)

    def copy(self):
        return StableSet(list(self._d))

    def clear(self):
        self._d = {}

    def __sub__(self, other):
        res = self.copy()
        for item in other:
            res.discard(item)
        return res

    def __str__(self) -> str:
        return str(list(self._d))

    def __repr__(self) -> str:
        return str(list(self._d))

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, value: Any) -> int:
        return value in self._d

    def __iter__(self):
        return list(self._d).__iter__()

    def _type_check(self, other):
        if not isinstance(other, StableSet):
            raise RuntimeError(
                f"A StableSet can only be operated with another StableSet! "
                f"Current type: {type(other)}."
            )

    def __eq__(self, other):
        self._type_check(other)
        return set(other._d) == set(self._d)

    def __le__(self, other):
        self._type_check(other)
        return set(self._d) <= set(other._d)

    def __lt__(self, other):
        self._type_check(other)
        return set(self._d) < set(other._d)

    def __ge__(self, other):
        self._type_check(other)
        return set(self._d) >= set(other._d)

    def __gt__(self, other):
        self._type_check(other)
        return set(self._d) > set(other._d)

    def __getitem__(self, idx):
        return list(self._d)[idx]
