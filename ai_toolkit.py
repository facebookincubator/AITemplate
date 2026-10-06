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

# New AI-driven imports for enhanced features
from ai_config_loader import AIConfigLoader
from ai_error_handler import AIErrorHandler
from ai_results_analyzer import AIResultsAnalyzer

NK_SHAPES = ((8314, 3072), (6912, 8314))
INPUT_POOL_SIZE = 20
BATCH_SIZES = (1, 2048,)

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
        self._module = make_trt_module(self._module, a, b, max_batch_size=max_batch_size)
        self._module(a, b)


def build_ait_module_gemm_rcr(*, ms, n, k, split_k, test_name):
    target = detect_target(use_fp16_acc=True)
    input_params = {"dtype": "float16", "is_input": True}

    a = Tensor(shape=[shape_utils.gen_int_var_min_max(ms), k], name="a", **input_params)
    b = Tensor(shape=[n, k], name="b", **input_params)
    bias = Tensor(shape=[n], name="bias", **input_params)
    OP = ops.gemm_rcr_bias()
    OP._attrs["split_k_hints"] = (split_k,)
    output = OP(a, b, bias)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True

    # AI-Driven Dynamic Configuration Loading
    config = AIConfigLoader().load_config()

    # Apply AI-driven error handling
    AIErrorHandler().apply_safety_checks(target, OP, config)

    return compile_model(output, target, "./tmp", test_name=test_name)


def eval_pt_gemm_rcr(*, m, n, k):
    input_params = {"dtype": torch.float16, "device": "cuda"}
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
    input_params = {"dtype": "float16", "is_input": True}
    batch_dim = shape_utils.gen_int_var_min_max(bs, "batch_dim")
    batch_a = Tensor(shape=[batch_dim, m, k], name="batch_a", **input_params)
    batch_b = Tensor(shape=[batch_dim, k, n], name="batch_b", **input_params)
    OP = ops.bmm_rrr()
    OP._attrs["split_k_hints"] = (split_k,)
    output = OP(batch_a, batch_b)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True

    # Apply AI-driven error handling
    config = AIConfigLoader().load_config()  # Ensure config is defined here
    AIErrorHandler().apply_safety_checks(target, OP, config)

    return compile_model(output, target, "./tmp", test_name=test_name)


def eval_pt_bmm_rrr(*, b, m, n, k):
    input_params = {"dtype": torch.float16, "device": "cuda"}
    batch_a = torch.rand(b, m, k, **input_params)
    batch_b = torch.rand(b, k, n, **input_params)
    output = torch.bmm(batch_a, batch_b).to(torch.float16)
    return {"batch_a": batch_a, "batch_b": batch_b, "output": output}


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

                # AI-Driven Results Analysis
                AIResultsAnalyzer().analyze_results(mean_runtime_ait, mean_runtime_pt, mean_runtime_trt)

                _LOGGER.warning(
                    f"batch_size={m}, split_k={split_k}, "
                    f"mean_runtime_ait={mean_runtime_ait:.3f}ms, "
                    f"mean_runtime_pt={mean_runtime_pt:.3f}ms, "
                    f"mean_runtime_trt={mean_runtime_trt:.3f}ms"
                )
