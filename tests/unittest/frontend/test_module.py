# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest
from typing import OrderedDict

import torch
import torch as pt
from aitemplate import frontend_v2 as ait

from aitemplate.compiler import ops


class NNModule(unittest.TestCase):
    def test_module(self):
        class AITModule(ait.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = ait.Parameter(shape=[32, 32], dtype="float16")
                self.b = ait.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        a = AITModule()
        ait_param_names = [x[0] for x in a.named_parameters()]

        b = PTModule()
        pt_param_names = [x[0] for x in b.named_parameters()]

        for (x, y) in zip(ait_param_names, pt_param_names):
            assert x == y

    def test_sequential_1(self):
        class AITModule(ait.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = ait.Parameter(shape=[32, 32], dtype="float16")
                self.b = ait.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        a = ait.nn.Sequential(AITModule(), AITModule(), AITModule())
        b = pt.nn.Sequential(
            PTModule(),
            PTModule(),
            PTModule(),
        )

        ait_param_names = [x[0] for x in a.named_parameters()]
        pt_param_names = [x[0] for x in b.named_parameters()]

        for (x, y) in zip(ait_param_names, pt_param_names):
            assert x == y

    def test_sequential_2(self):
        class AITModule(ait.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = ait.Parameter(shape=[32, 32], dtype="float16")
                self.b = ait.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        a = ait.nn.Sequential(
            OrderedDict(
                [
                    ("block1", AITModule()),
                    ("block2", AITModule()),
                    ("block3", AITModule()),
                ]
            )
        )
        b = pt.nn.Sequential(
            OrderedDict(
                [
                    ("block1", PTModule()),
                    ("block2", PTModule()),
                    ("block3", PTModule()),
                ]
            )
        )

        ait_param_names = [x[0] for x in a.named_parameters()]
        pt_param_names = [x[0] for x in b.named_parameters()]

        for (x, y) in zip(ait_param_names, pt_param_names):
            assert x == y

    def test_module_dict(self):
        class AITModule(ait.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = ait.Parameter(shape=[32, 32], dtype="float16")
                self.b = ait.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        class AITDict(ait.nn.Module):
            def __init__(self):
                super().__init__()
                self.dict_a = ait.nn.ModuleDict(
                    {
                        "block1": AITModule(),
                        "block2": AITModule(),
                    }
                )
                self.dict_b = ait.nn.ModuleDict(
                    {
                        "block_a": AITModule(),
                        "block_b": AITModule(),
                    }
                )

            def forward(self, x):
                return (
                    self.dict_a["block1"](x)
                    + self.dict_a["block2"](x)
                    + self.dict_b["block_a"](x)
                    + self.dict_b["block_b"](x)
                )

        class PTDict(pt.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dict_a = pt.nn.ModuleDict(
                    {
                        "block1": PTModule(),
                        "block2": PTModule(),
                    }
                )
                self.dict_b = pt.nn.ModuleDict(
                    {
                        "block_a": PTModule(),
                        "block_b": PTModule(),
                    }
                )

            def forward(self, x):
                return (
                    self.dict_a["block1"](x)
                    + self.dict_a["block2"](x)
                    + self.dict_b["block_a"](x)
                    + self.dict_b["block_b"](x)
                )

        a = AITDict()
        b = PTDict()

        ait_param_names = [x[0] for x in a.named_parameters()]
        pt_param_names = [x[0] for x in b.named_parameters()]

        for (x, y) in zip(ait_param_names, pt_param_names):
            assert x == y


if __name__ == "__main__":
    unittest.main()
