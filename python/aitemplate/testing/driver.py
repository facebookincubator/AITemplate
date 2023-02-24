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
import random
import unittest

import torch

from aitemplate.testing import detect_target


class AITTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(AITTestCase, self).__init__(*args, **kwargs)
        self.target = detect_target()
        self._renew_id()

    def _renew_id(self):
        self._test_id = "%06x" % random.randrange(16**6)

    def _check_dtype(self, dtype):
        if self.target.name() == "rocm" and dtype != "float16":
            self.skipTest(f"Rocm doesn't support {dtype}")
        if (
            self.target.name() == "cuda"
            and dtype == "bfloat16"
            and int(self.target._arch) < 80
        ):
            self.skipTest(f"CUDA SM{self.target._arch} doesn't support {dtype}")
