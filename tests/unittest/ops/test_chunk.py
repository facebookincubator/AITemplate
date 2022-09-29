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
import logging
import unittest
from typing import List

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
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

        module = compile_model(Ys, target, "./tmp", "chunk")

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

            module.run_with_tensors([x_pt], outputs)

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
