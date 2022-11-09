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

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils import shape_utils


class ConcatenateTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ConcatenateTestCase, self).__init__(*args, **kwargs)

    def _run_concatenate(
        self, *, concatenate_op, input_shapes, dim=None, input_type="float16"
    ):
        logging.info(
            "Test input shapes {input_shapes}, dim={dim}".format(
                input_shapes=input_shapes, dim=dim
            )
        )

        # generate torch reference result
        input_tensors_pt = [
            get_random_torch_tensor(shape, input_type)
            for i, shape in enumerate(input_shapes)
        ]
        Y_pt = (
            torch.cat(input_tensors_pt)
            if dim is None
            else torch.cat(input_tensors_pt, dim)
        )
        Y_pt = torch.tanh(Y_pt)

        target = detect_target()
        inputs = [
            Tensor(
                shape=shape, dtype=input_type, name="input_{}".format(i), is_input=True
            )
            for i, shape in enumerate(input_shapes)
        ]
        Y = concatenate_op(inputs) if dim is None else concatenate_op(inputs, dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_shape = [d._attrs["values"][0] for d in Y._attrs["shape"]]

        logging.info("AITemplate output_shape: {}".format(y_shape))

        module = compile_model(Y, target, "./tmp", "concatenate_tanh")

        input_tensors_ait = {
            f"input_{idx}": input_tensors_pt[idx] for idx in range(len(inputs))
        }
        y = torch.empty(y_shape).cuda().half()
        module.run_with_tensors(input_tensors_ait, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _run_batch_concatenate(
        self, *, batch_sizes, concatenate_op, input_shapes, dim=0, input_type="float16"
    ):
        logging.info(
            "Batch test input shapes {input_shapes}, dim={dim}".format(
                input_shapes=input_shapes, dim=dim
            )
        )
        batch_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_size")
        target = detect_target()
        inputs = [
            Tensor(
                shape=[
                    batch_dim,
                    *shape,
                ],
                dtype=input_type,
                name="input_{}".format(i),
                is_input=True,
            )
            for i, shape in enumerate(input_shapes)
        ]
        Y = concatenate_op(inputs) if dim is None else concatenate_op(inputs, dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        batch_tag = "_".join([str(b) for b in batch_sizes])
        module = compile_model(Y, target, "./tmp", f"concatenate_tanh_{batch_tag}")
        for batch in batch_sizes:
            logging.info("checking batch: {}".format(batch))
            input_tensors_pt = [
                get_random_torch_tensor([batch, *shape], input_type)
                for i, shape in enumerate(input_shapes)
            ]
            Y_pt = (
                torch.cat(input_tensors_pt)
                if dim is None
                else torch.cat(input_tensors_pt, dim)
            )
            Y_pt = torch.tanh(Y_pt)

            input_tensors_ait = {
                f"input_{idx}": input_tensors_pt[idx] for idx in range(len(inputs))
            }
            y = torch.empty_like(Y_pt)
            module.run_with_tensors(input_tensors_ait, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _run_masked_concatenate(
        self,
        *,
        concatenate_op,
        input_shapes,
        input_masks,
        dim=None,
        input_type="float16",
    ):
        logging.info(
            "Test input shapes {input_shapes}, input_masks={input_masks}, dim={dim}".format(
                input_shapes=input_shapes, input_masks=input_masks, dim=dim
            )
        )

        # generate torch reference result
        input_tensors_pt = [
            get_random_torch_tensor(shape, input_type)
            for i, shape in enumerate(input_shapes)
        ]
        Y_pt = (
            torch.tanh(torch.cat(input_tensors_pt))
            if dim is None
            else torch.tanh(torch.cat(input_tensors_pt, dim))
        )
        y_pt = Y_pt.cpu().numpy()

        target = detect_target()
        inputs = [
            Tensor(
                shape=shape, dtype=input_type, name="input_{}".format(i), is_input=True
            )
            for i, shape in enumerate(input_shapes)
        ]
        Y = concatenate_op(inputs) if dim is None else concatenate_op(inputs, dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        y_shape = [d._attrs["values"][0] for d in Y._attrs["shape"]]

        # setup new input_masks, inputs and input_accessors
        inputs = [i for mask, i in zip(input_masks, inputs) if mask is True]
        input_accessors = [
            i
            for mask, i in zip(input_masks, concatenate_op._attrs["input_accessors"])
            if mask is True
        ]
        concatenate_op._attrs["input_masks"] = input_masks
        concatenate_op._attrs["inputs"] = inputs
        concatenate_op._attrs["input_accessors"] = input_accessors

        logging.info("AITemplate output_shape: {}".format(y_shape))

        module = compile_model(Y, target, "./tmp", "concatenate_tanh")

        inputs = []
        for i, x_tensor_pt in enumerate(input_tensors_pt):
            if input_masks[i]:
                inputs.append(x_tensor_pt)
        y = torch.empty(y_shape).cuda().half()
        module.run_with_tensors(inputs, [y])

        split_sections = []
        split_offset = 0
        for shape in input_shapes[:-1]:
            split_offset = split_offset + shape[dim]
            split_sections.append(split_offset)

        ys_pt = np.split(y_pt, split_sections, axis=dim)
        ys = np.split(y.cpu().numpy(), split_sections, axis=dim)
        for mask, pt, actual in zip(input_masks, ys_pt, ys):
            if mask is True:
                np.testing.assert_allclose(actual, pt, atol=1e-2, rtol=1e-2)

    def test_batch_cat(self):
        self._run_batch_concatenate(
            batch_sizes=[1, 1],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([1], [1]),
            dim=0,
        )
        self._run_batch_concatenate(
            batch_sizes=[1, 1],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([1], [1]),
            dim=1,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=0,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=1,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=2,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=3,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 1, 4], [2, 3, 4]),
            dim=2,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 2]),
            dim=3,
        )

    def test_cat(self):
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(), input_shapes=([1], [1]), dim=0
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(), input_shapes=([1, 1], [1, 1]), dim=0
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(), input_shapes=([1, 1], [1, 1]), dim=1
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(), input_shapes=([2, 1], [2, 1]), dim=1
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(), input_shapes=[[2, 3, 4]], dim=1
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=0,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=2,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [3, 3, 4], [4, 3, 4]),
            dim=0,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 4, 4], [2, 5, 4]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 6], [2, 3, 5], [2, 3, 4]),
            dim=2,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([1024, 32, 32], [1024, 16, 32], [1024, 8, 32]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([12, 3, 4, 5], [3, 3, 4, 5], [7, 3, 4, 5]),
            dim=0,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 9, 5], [2, 3, 4, 5], [2, 3, 1, 5]),
            dim=2,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4, 5], [2, 3, 4, 3], [2, 3, 4, 5]),
            dim=3,
        )

        self._run_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([1, 3, 1], [2, 3, 1], [3, 3, 1]),
        )

        # self._run_concatenate(concatenate_op=ops.concatenate(),
        #                       input_shapes=([12, 3, 4, 5], [3, 3, 4, 5], [7, 3, 4, 5]), dim=0)
        # self._run_concatenate(concatenate_op=ops.concatenate(),
        #                       input_shapes=([2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]), dim=1)
        # self._run_concatenate(concatenate_op=ops.concatenate(),
        #                       input_shapes=([2, 3, 9, 5], [2, 3, 4, 5], [2, 3, 1, 5]), dim=2)
        # self._run_concatenate(concatenate_op=ops.concatenate(),
        #                       input_shapes=([2, 3, 4, 5], [2, 3, 4, 3], [2, 3, 4, 5]), dim=3)
        # self._run_concatenate(concatenate_op=ops.concatenate(),
        #                       input_shapes=([1, 3, 1], [2, 3, 1], [3, 3, 1]))

    def test_masked_cat(self):
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2], [2]),
            input_masks=[True, False],
            dim=0,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3], [5, 3], [3, 3]),
            input_masks=[False, True, True],
            dim=0,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 11, 4], [2, 5, 4], [2, 2, 4]),
            input_masks=[True, False, True],
            dim=1,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([1, 1, 1], [1, 1, 2], [1, 1, 4]),
            input_masks=[False, True, False],
            dim=2,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate_tanh(),
            input_shapes=([2, 3, 4], [2, 3, 8], [2, 3, 16]),
            input_masks=[False, True, False],
            dim=2,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
