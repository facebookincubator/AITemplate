# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest
from typing import List

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.testing.test_utils import get_random_torch_tensor


class ChunkTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ChunkTestCase, self).__init__(*args, **kwargs)

    def _run_chunk(
        self,
        *,
        input_shape: List[IntVar],
        chunks: int,
        dim: int = 0,
        input_type="float16",
    ):
        logging.info(f"{input_shape=}, " f"{chunks=}, " f"{dim=}")

        chunk_op = ops.chunk()
        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)
        Ys = chunk_op(X, chunks, dim)
        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = "output_{}".format(idx)
            Y._attrs["is_output"] = True

        module = gen_execution_module(Ys, target, "./tmp", "chunk")

        for batch_size in input_shape[0]._attrs["values"]:
            logging.info(f"Testing {batch_size=}")
            x_pt = get_random_torch_tensor(
                [batch_size, *[v.value() for v in input_shape[1:]]], input_type
            )
            ys_pt = torch.chunk(x_pt, chunks, dim)
            y_shapes = [Y_pt.size() for Y_pt in ys_pt]
            outputs = {
                f"output_{idx}": torch.empty(y_shape).cuda().half()
                for idx, y_shape in enumerate(y_shapes)
            }

            module.RunWithTensors([x_pt], outputs)

            for idx, y_pt in enumerate(ys_pt):
                self.assertTrue(
                    torch.allclose(y_pt, outputs[f"output_{idx}"], atol=1e-2, rtol=1e-2)
                )

    def test_chunk(self):
        self._run_chunk(
            input_shape=[IntImm(17), IntImm(5), IntImm(29)],
            chunks=2,
            dim=0,
            input_type="float16",
        )
        self._run_chunk(
            input_shape=[IntImm(17), IntImm(5), IntImm(29)],
            chunks=7,
            dim=1,
            input_type="float16",
        )
        self._run_chunk(
            input_shape=[IntImm(17), IntImm(5), IntImm(29)],
            chunks=11,
            dim=2,
            input_type="float16",
        )

    def test_dynamic_chunk(self):
        self._run_chunk(
            input_shape=[
                IntVar(values=[13, 17], name="batch_dim"),
                IntImm(5),
                IntImm(29),
            ],
            chunks=2,
            dim=1,
            input_type="float16",
        )
        with self.assertRaises(RuntimeError) as context:
            self._run_chunk(
                input_shape=[
                    IntVar(values=[13, 17], name="batch_dim"),
                    IntImm(5),
                    IntImm(29),
                ],
                chunks=2,
                dim=0,
                input_type="float16",
            )
        self.assertTrue(
            "Not implemented: chunk along dynamic axes" in str(context.exception)
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
