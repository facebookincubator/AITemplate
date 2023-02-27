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

import os
import unittest

import torch

from .demo import run_model

try:
    from libfb.py.asyncio.await_utils import await_sync
    from manifold.clients.python import ManifoldClient
except ImportError:
    ManifoldClient = None


class BertBaseUncasedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def test_bert_base_uncased(self):
        model_path = "bert-base-uncased"
        if ManifoldClient is not None:
            model_path = "/tmp/aitemplate_bert/bert-base-uncased"
            os.makedirs(model_path, exist_ok=True)
            with ManifoldClient.get_client(bucket="glow_test_data") as client:
                await_sync(
                    client.getRecursive(
                        manifold_path="tree/aitemplate/bert/bert-base-uncased",
                        local_path=model_path,
                    )
                )
        run_model(
            prompt="The quick brown fox jumps over the lazy dog.",
            activation="fast_gelu",
            graph_mode=True,
            use_fp16_acc=True,
            verify=True,
            model_path=model_path,
        )


if __name__ == "__main__":
    unittest.main()
