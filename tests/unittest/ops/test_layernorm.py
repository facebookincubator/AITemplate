# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for LayerNorm Operator.
"""
import logging
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


class LayernormTestCase(unittest.TestCase):
    def _test_layernorm(
        self,
        MS=(),
        NS=(1496,),
        gamma_is_none=False,
        beta_is_none=False,
        use_size_op=False,
        eps=1e-5,
    ):
        BS = [1, 1024]
        input_shapes = ((BS), *MS, *NS)
        logging.info(
            f"input shapes: {input_shapes}"
            f"gamma_is_none: {gamma_is_none}, beta_is_none: {beta_is_none}, "
            f"use_size_op: {use_size_op}"
        )
        assert isinstance(MS, (list, tuple))
        assert isinstance(NS, (list, tuple))

        X1 = Tensor(
            shape=[IntVar(name="input_batch", values=BS), *MS, *NS],
            dtype="float16",
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=NS,
                dtype="float16",
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=NS,
                dtype="float16",
                name="beta",
                is_input=True,
            )
        if use_size_op:
            norm_shapes = [
                ops.getitem()(ops.size()(X1), i) for i in range(1 + len(MS), X1._rank())
            ]
        else:
            norm_shapes = [IntImm(n) for n in NS]

        X4 = (
            ops.layernorm()(X1, X2, X3, NS, eps)
            if not use_size_op
            else ops.layernorm()(X1, X2, X3, norm_shapes, eps)
        )
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = gen_execution_module(X4, target, "./tmp", "layernorm")

        for batch_size in [50, 900, 1024]:
            x1_pt = torch.randn(batch_size, *MS, *NS).cuda().half()
            if gamma_is_none:
                x2_pt = None
            else:
                x2_pt = torch.randn(NS).cuda().half()
            if beta_is_none:
                x3_pt = None
            else:
                x3_pt = torch.randn(NS).cuda().half()
            x4_pt = torch.nn.functional.layer_norm(x1_pt, NS, x2_pt, x3_pt, eps=eps)

            inputs = {"X": x1_pt}
            if not gamma_is_none:
                inputs["gamma"] = x2_pt
            if not beta_is_none:
                inputs["beta"] = x3_pt
            x4 = torch.empty([batch_size, *MS, *NS]).cuda().half()
            module.RunWithTensors(inputs, [x4])
            self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-3, rtol=1e-3))

    def test_layernorm(self):
        if detect_target().name() == "rocm":
            self._test_layernorm(use_size_op=False, MS=(256,), NS=(768,))
        else:
            for use_size_op in (True, False):
                self._test_layernorm(use_size_op=use_size_op)
                self._test_layernorm(gamma_is_none=True, use_size_op=use_size_op)
                self._test_layernorm(beta_is_none=True, use_size_op=use_size_op)
                self._test_layernorm(
                    gamma_is_none=True, beta_is_none=True, use_size_op=use_size_op
                )
                self._test_layernorm(use_size_op=use_size_op, eps=0.1)
                self._test_layernorm(MS=(16, 64), NS=(4, 32), use_size_op=use_size_op)
                self._test_layernorm(
                    MS=(16, 8, 4), NS=(2, 4, 32), use_size_op=use_size_op
                )


if __name__ == "__main__":
    unittest.main()
