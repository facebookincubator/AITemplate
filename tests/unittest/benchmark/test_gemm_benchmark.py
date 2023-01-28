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

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import Tensor
from aitemplate.compiler.compiler import compile_model

from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_ait import make_input_output_pools, run_benchmark
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.testing.benchmark_trt import make_trt_module
from aitemplate.utils import shape_utils

NK_SHAPES = ((8314, 3072), (6912, 8314))
INPUT_POOL_SIZE = 20
BATCH_SIZES = (
    1,
    2048,
)


_LOGGER = logging.getLogger(__name__)


class GemmRCRModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.nn.functional.linear(a, b)


class GemmRCRFunction:
    def __init__(self, inputs_pool):
        self._it_pool = 0
        self._as = [t["a"] for t in inputs_pool]
        self._bs = [t["b"] for t in inputs_pool]
        self._inputs_pool_size = len(inputs_pool)
        self._module = GemmRCRModule()

    def next_input(self):
        self._it_pool += 1
        self._it_pool %= self._inputs_pool_size
        return self._as[self._it_pool], self._bs[self._it_pool]

    def __call__(self):
        return self._module(*self.next_input())


class GemmRCRTRTFunction(GemmRCRFunction):
    def __init__(self, inputs_pool, max_batch_size):
        super().__init__(inputs_pool)
        a, b = self.next_input()
        self._module = make_trt_module(
            self._module, a, b, max_batch_size=max_batch_size
        )
        self._module(a, b)


def build_ait_module_gemm_rcr(*, ms, n, k, split_k, test_name):
    target = detect_target(use_fp16_acc=True)
    input_params = {
        "dtype": "float16",
        "is_input": True,
    }
    a = Tensor(shape=[shape_utils.gen_int_var_min_max(ms), k], name="a", **input_params)
    b = Tensor(shape=[n, k], name="b", **input_params)
    bias = Tensor(shape=[n], name="bias", **input_params)
    OP = ops.gemm_rcr_bias()
    OP._attrs["split_k_hints"] = (split_k,)
    output = OP(a, b, bias)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True
    return compile_model(output, target, "./tmp", test_name=test_name)


def eval_pt_gemm_rcr(*, m, n, k):
    input_params = {
        "dtype": torch.float16,
        "device": "cuda",
    }
    a = torch.rand(m, k, **input_params)
    b = torch.rand(n, k, **input_params)
    bias = torch.rand(n, **input_params)
    output = torch.nn.functional.linear(a, b, bias).to(torch.float16)
    return {"a": a, "b": b, "bias": bias, "output": output}


class BmmRRRModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.bmm(a, b)


class BmmRRRFunction:
    def __init__(self, inputs_pool):
        self._it_pool = 0
        self._as = [t["batch_a"] for t in inputs_pool]
        self._bs = [t["batch_b"] for t in inputs_pool]
        self._inputs_pool_size = len(inputs_pool)
        self._module = BmmRRRModule()

    def next_input(self):
        self._it_pool += 1
        self._it_pool %= self._inputs_pool_size
        return self._as[self._it_pool], self._bs[self._it_pool]

    def __call__(self):
        return self._module(*self.next_input())


class BmmRRRTRTFunction(BmmRRRFunction):
    def __init__(self, inputs_pool, max_batch_size):
        super().__init__(inputs_pool)
        batch_as, batch_bs = self.next_input()
        self._module = make_trt_module(
            self._module, batch_as, batch_bs, max_batch_size=max_batch_size
        )
        self._module(batch_as, batch_bs)


def build_ait_module_bmm_rrr(*, bs, m, n, k, split_k, test_name):
    target = detect_target(use_fp16_acc=True)
    input_params = {
        "dtype": "float16",
        "is_input": True,
    }
    batch_dim = shape_utils.gen_int_var_min_max(bs, "batch_dim")
    batch_a = Tensor(
        shape=[batch_dim, m, k],
        name="batch_a",
        **input_params,
    )
    batch_b = Tensor(
        shape=[batch_dim, k, n],
        name="batch_b",
        **input_params,
    )
    OP = ops.bmm_rrr()
    OP._attrs["split_k_hints"] = (split_k,)
    output = OP(batch_a, batch_b)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True
    return compile_model(output, target, "./tmp", test_name=test_name)


def eval_pt_bmm_rrr(*, b, m, n, k):
    input_params = {
        "dtype": torch.float16,
        "device": "cuda",
    }
    batch_a = torch.rand(b, m, k, **input_params)
    batch_b = torch.rand(b, k, n, **input_params)
    output = torch.bmm(batch_a, batch_b).to(torch.float16)
    return {
        "batch_a": batch_a,
        "batch_b": batch_b,
        "output": output,
    }


class TestGemmRCRBenchmark(unittest.TestCase):
    @unittest.skipIf(
        detect_target(use_fp16_acc=True).in_ci_env(), "don't run benchmark in CI"
    )
    def test_benchmark(self):
        split_ks = sorted(set(range(1, 6)).union([2**i for i in range(5)]))
        for split_k, (n, k) in itertools.product(split_ks, NK_SHAPES):
            NUM_ITERS = 100000
            NUM_WARMUP_ITERS = 1000
            ait_module = build_ait_module_gemm_rcr(
                ms=BATCH_SIZES,
                n=n,
                k=k,
                split_k=split_k,
                test_name=f"gemm_rcr_{split_k=}_{uuid.uuid4().hex}",
            )
            for m in BATCH_SIZES:
                mnk = {"m": m, "n": n, "k": k}
                _LOGGER.warning(f"mnk={mnk}, split_k={split_k}")
                inputs_pool, outputs_pool = make_input_output_pools(
                    pool_size=INPUT_POOL_SIZE,
                    eval_pt_func=lambda: eval_pt_gemm_rcr(**mnk),
                    input_filter_func=lambda name, _: not name.startswith("output"),
                    output_filter_func=lambda name, _: name.startswith("output"),
                )
                gemm_rcr_function = GemmRCRFunction(inputs_pool)
                gemm_rcr_trt_function = GemmRCRTRTFunction(
                    inputs_pool, max_batch_size=m
                )

                pt_outputs = eval_pt_gemm_rcr(**mnk)
                ait_outputs = {"output": torch.empty_like(pt_outputs["output"])}
                ait_module.run_with_tensors(
                    {k: v for k, v in pt_outputs.items() if k != "output"},
                    ait_outputs,
                )
                torch.testing.assert_close(
                    ait_outputs["output"], pt_outputs["output"], rtol=1, atol=1
                )
                mean_runtime_ait = run_benchmark(
                    ait_module=ait_module,
                    inputs_pool=inputs_pool,
                    outputs_pool=outputs_pool,
                    num_iters=NUM_ITERS,
                    num_warmup_iters=NUM_WARMUP_ITERS,
                )

                mean_runtime_pt = benchmark_torch_function(
                    iters=NUM_ITERS, function=gemm_rcr_function
                )

                mean_runtime_trt = benchmark_torch_function(
                    iters=NUM_ITERS, function=gemm_rcr_trt_function
                )

                benchmark_results = {
                    "function": "gemm_rcr_bias",
                    "mean_runtime_ait_ms": round(mean_runtime_ait, 5),
                    "mean_runtime_pt_ms": round(mean_runtime_pt, 5),
                    "mean_runtime_trt_ms": round(mean_runtime_trt, 5),
                    "split_k": split_k,
                    **mnk,
                }
                _LOGGER.warning(
                    f"Benchmark results {json.dumps(benchmark_results, separators=(',', ':'))}",
                )


class TestBmmRRRBenchmark(unittest.TestCase):
    @unittest.skipIf(
        detect_target(use_fp16_acc=True).in_ci_env(), "don't run benchmark in CI"
    )
    def test_benchmark(self):
        INPUT_POOL_SIZE = 3
        MNK_SHAPES = ((1469, 16, 128),)
        split_ks = sorted(set(range(1, 6)).union([2**i for i in range(5)]))
        for split_k, (m, n, k) in itertools.product(split_ks, MNK_SHAPES):
            NUM_ITERS = 100000
            NUM_WARMUP_ITERS = 1000
            ait_module = build_ait_module_bmm_rrr(
                bs=BATCH_SIZES,
                m=m,
                n=n,
                k=k,
                split_k=split_k,
                test_name=f"bmm_rrr_{split_k=}_{uuid.uuid4().hex}",
            )
            for b in BATCH_SIZES:
                bmnk = {"b": b, "m": m, "n": n, "k": k}
                _LOGGER.warning(f"bmnk={bmnk}, split_k={split_k}")
                inputs_pool, outputs_pool = make_input_output_pools(
                    pool_size=INPUT_POOL_SIZE,
                    eval_pt_func=lambda: eval_pt_bmm_rrr(**bmnk),
                    input_filter_func=lambda name, _: not name.startswith("output"),
                    output_filter_func=lambda name, _: name.startswith("output"),
                )

                bmm_rrr_function = BmmRRRFunction(inputs_pool)
                bmm_rrr_trt_function = BmmRRRTRTFunction(inputs_pool, max_batch_size=b)

                pt_outputs = eval_pt_bmm_rrr(**bmnk)
                ait_outputs = {"output": torch.empty_like(pt_outputs["output"])}
                ait_module.run_with_tensors(
                    {k: v for k, v in pt_outputs.items() if k != "output"},
                    ait_outputs,
                )
                torch.testing.assert_close(
                    ait_outputs["output"], pt_outputs["output"], rtol=1, atol=1
                )

                mean_runtime_ait = run_benchmark(
                    ait_module=ait_module,
                    inputs_pool=inputs_pool,
                    outputs_pool=outputs_pool,
                    num_iters=NUM_ITERS,
                    num_warmup_iters=NUM_WARMUP_ITERS,
                )

                mean_runtime_pt = benchmark_torch_function(
                    iters=NUM_ITERS, function=bmm_rrr_function
                )

                mean_runtime_trt = benchmark_torch_function(
                    iters=NUM_ITERS, function=bmm_rrr_trt_function
                )

                benchmark_results = {
                    "function": "bmm_rrr",
                    "mean_runtime_ait_ms": round(mean_runtime_ait, 5),
                    "mean_runtime_pt_ms": round(mean_runtime_pt, 5),
                    "mean_runtime_trt_ms": round(mean_runtime_trt, 5),
                    "split_k": split_k,
                    **bmnk,
                }
                _LOGGER.warning(
                    f"Benchmark results {json.dumps(benchmark_results, separators=(',', ':'))}",
                )


if __name__ == "__main__":
    unittest.main()
