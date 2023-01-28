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

import itertools
import json
import logging
import unittest
import uuid

from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_ait import make_input_output_pools, run_benchmark

from ..compiler.test_strided_layernorm import build_ait_module, eval_pt

LOGGER = logging.getLogger(__name__)


class TestStridedLayerNormBenchmark(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_id = 0

    @unittest.skipIf(detect_target().in_ci_env(), "don't run benchmark in CI")
    def test_benchmark(self):
        for (input_nonbatch_shape, (start_indices, end_indices),) in itertools.product(
            ((2048, 256), (2048, 512), (2048, 1024), (2048, 2048)),
            (((0, 0, 4), (None, None, 224)), ((0, 0, 3), (None, None, 223))),
        ):
            BATCH_SIZE = 2
            NUM_ITERS = 1000000
            NUM_WARMUP_ITERS = 10000
            INPUT_POOL_SIZE = 100
            _layernorm_common_params = {
                "input_nonbatch_shape": input_nonbatch_shape,
                "n_normalize_over_last_dims": 1,
                "gamma_is_none": None,
                "beta_is_none": None,
                "fuse_sigmoid_mul": False,
                "eps": 1e-5,
                "start_indices": start_indices,
                "end_indices": end_indices,
            }
            ait_module = build_ait_module(
                batch_sizes=(BATCH_SIZE,),
                workdir=uuid.uuid4().hex,
                test_id=self.test_id,
                **_layernorm_common_params,
            )
            self.test_id += 1
            inputs_pool, outputs_pool = make_input_output_pools(
                pool_size=INPUT_POOL_SIZE,
                eval_pt_func=lambda: eval_pt(
                    batch_size=BATCH_SIZE,
                    **_layernorm_common_params,
                ),
                input_filter_func=lambda k, v: not k.startswith("output")
                and v is not None,
                output_filter_func=lambda k, _: k.startswith("output"),
            )
            mean_runtime = run_benchmark(
                ait_module=ait_module,
                inputs_pool=inputs_pool,
                outputs_pool=outputs_pool,
                num_iters=NUM_ITERS,
                num_warmup_iters=NUM_WARMUP_ITERS,
            )
            benchmark_results = {
                "mean_runtime": mean_runtime,
                "input_nonbatch_shape": input_nonbatch_shape,
                "start_indices": start_indices,
                "end_indices": end_indices,
            }
            LOGGER.warning(
                f"Benchmark results {json.dumps(benchmark_results, separators=(',', ':'))}"
            )


if __name__ == "__main__":
    unittest.main()
