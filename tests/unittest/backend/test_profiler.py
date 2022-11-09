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

import unittest
from random import randrange

from aitemplate.backend import profiler_runner

profiler_runner.extract_profile_result = lambda _: (
    "",
    False,
)

from time import sleep

from aitemplate.backend.cuda.target_def import CUDA as CUDATarget

from aitemplate.backend.profiler_runner import ProfilerRunner


def dice():
    return randrange(1, 10) / 4


class Delegate:
    def __init__(self, test_instance):
        self._test_instance = test_instance
        self.results = [None] * 42

    def add_result(self, idx, val):
        sleep(dice())
        self.results[idx] = val

    def postprocess_results(self):
        for i, val in enumerate(self.results):
            self._test_instance.assertTrue(val is not None, f"Result {i} not filled in")


def delegate_cb_wrapper(idx, value):
    def wrapped(result, delegate):
        return delegate.add_result(idx, value)

    return wrapped


class ProfilerTestCase(unittest.TestCase):
    def test_profiler_runner(self):
        with CUDATarget() as _:
            pr = ProfilerRunner(
                devices=[str(i) for i in range(12)],
                timeout=60,
                postprocessing_delegate=Delegate(test_instance=self),
            )

            for i, _ in enumerate(pr._postprocessing_delegate.results):
                sleep_for = 0
                pr.push(
                    cmds=["sleep", f"{sleep_for}"],
                    process_result_callback=delegate_cb_wrapper(i, sleep_for),
                )

            pr.join()


if __name__ == "__main__":
    unittest.main()
