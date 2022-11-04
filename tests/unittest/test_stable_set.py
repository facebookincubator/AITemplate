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
Unittests for StableSet.
"""
import unittest

from aitemplate.compiler.stable_set import StableSet


class StableSetTestCase(unittest.TestCase):
    def test_stable_set(self):
        s = StableSet([5, 2, 1])

        s.add(4)
        s.add(5)
        s.add(1)
        self.assertEqual(s, StableSet([5, 2, 1, 4]))

        s.discard(4)
        s.discard(4)
        s.discard(10)
        self.assertEqual(s, StableSet([5, 2, 1]))

        s.remove(1)
        self.assertEqual(s, StableSet([5, 2]))
        with self.assertRaises(KeyError):
            s.remove(1)

        s.update([1, 5, 9])
        self.assertEqual(s, StableSet([5, 2, 1, 9]))

        s1 = s.copy()
        self.assertEqual(s, s1)
        self.assertNotEqual(id(s._d), id(s1._d))

        s1 = s - [1]
        self.assertEqual(s1, StableSet([5, 2, 9]))
        self.assertEqual(s, StableSet([5, 2, 1, 9]))

        self.assertEqual(len(s), 4)

        self.assertTrue(1 in s)
        self.assertTrue(1 not in s1)

        s1 = list(s)
        self.assertEqual(s, StableSet(s1))

        self.assertTrue(s >= StableSet([5, 2, 1, 9]))
        self.assertTrue(s > StableSet([5, 1, 2]))
        self.assertTrue(s <= StableSet([5, 2, 1, 9]))
        self.assertTrue(s < StableSet([5, 2, 1, 9, 10]))


if __name__ == "__main__":
    unittest.main()
