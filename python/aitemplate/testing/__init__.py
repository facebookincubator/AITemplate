# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] testing module
"""
from . import benchmark_pt, benchmark_trt
from .detect_target import detect_target
from .gen_test_module import gen_execution_module
from .model import AITemplateTensor, Model

__all__ = [
    "benchmark_pt",
    "benchmark_trt",
    "detect_target",
    "gen_execution_module",
    "Model",
    "AITemplateTensor",
]
