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
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils import shape_utils
from parameterized import parameterized


class ConcatenateTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ConcatenateTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _run_concatenate(
        self, *, concatenate_op, input_shapes, dim=None, input_type="float16"
    ):
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

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", "concatenate", dll_name=dll_name)

        input_tensors_ait = {
            f"input_{idx}": input_tensors_pt[idx] for idx in range(len(inputs))
        }
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(input_tensors_ait, [y])
        self.assertTrue(torch.equal(Y_pt, y))

        self.test_count += 1

    def _run_batch_concatenate(
        self, *, batch_sizes, concatenate_op, input_shapes, dim=0, input_type="float16"
    ):
        target = detect_target()
        BATCH_DIM_NAME = "input_batch"
        batch_dim = shape_utils.gen_int_var_min_max(
            values=batch_sizes, name=BATCH_DIM_NAME
        )
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
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", f"concatenate_batched_{batch_tag}", dll_name=dll_name
        )
        for batch in batch_sizes:
            input_tensors_pt = [
                get_random_torch_tensor([batch, *shape], input_type)
                for i, shape in enumerate(input_shapes)
            ]
            Y_pt = (
                torch.cat(input_tensors_pt)
                if dim is None
                else torch.cat(input_tensors_pt, dim)
            )
            input_tensors_ait = {
                f"input_{idx}": input_tensors_pt[idx] for idx in range(len(inputs))
            }
            y = torch.empty_like(Y_pt)
            module.run_with_tensors(input_tensors_ait, [y])
            self.assertTrue(torch.equal(Y_pt, y))
            self.test_count += 1

    def _run_masked_concatenate(
        self,
        *,
        concatenate_op,
        input_shapes,
        input_masks,
        dim=None,
        input_type="float16",
    ):
        # generate torch reference result
        input_tensors_pt = [
            get_random_torch_tensor(shape, input_type)
            for i, shape in enumerate(input_shapes)
        ]
        y_pt = (
            torch.cat(input_tensors_pt)
            if dim is None
            else torch.cat(input_tensors_pt, dim)
        )

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

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", "concatenate_masked", dll_name=dll_name
        )

        inputs = []
        for i, x_tensor_pt in enumerate(input_tensors_pt):
            if input_masks[i]:
                inputs.append(x_tensor_pt)
        y = torch.empty_like(y_pt)
        module.run_with_tensors(inputs, [y])

        split_sections = [shape[dim] for shape in input_shapes]

        ys_pt = torch.split(y_pt, split_sections, dim=dim)
        ys = torch.split(y, split_sections, dim=dim)
        for mask, pt, actual in zip(input_masks, ys_pt, ys):
            if mask is True:
                self.assertTrue(torch.equal(pt, actual))
        self.test_count += 1

    def test_batch_cat(self):
        self._run_batch_concatenate(
            batch_sizes=[1, 1],
            concatenate_op=ops.concatenate(),
            input_shapes=([1], [1]),
            dim=0,
        )
        self._run_batch_concatenate(
            batch_sizes=[1, 1],
            concatenate_op=ops.concatenate(),
            input_shapes=([1], [1]),
            dim=1,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=0,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=1,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=2,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [2, 3, 4]),
            dim=3,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 1, 4], [2, 3, 4]),
            dim=2,
        )

    def test_cat(self):
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([1], [1]), dim=0
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([1, 1], [1, 1]), dim=0
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([1, 1], [1, 1]), dim=1
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([2, 1], [2, 1]), dim=1
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=[[2, 3, 4]], dim=1
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([2, 3, 4], [2, 3, 4]), dim=0
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([2, 3, 4], [2, 3, 4]), dim=1
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([2, 3, 4], [2, 3, 4]), dim=2
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [3, 3, 4], [4, 3, 4]),
            dim=0,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [2, 4, 4], [2, 5, 4]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 6], [2, 3, 5], [2, 3, 4]),
            dim=2,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([1024, 32, 32], [1024, 16, 32], [1024, 8, 32]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([12, 3, 4, 5], [3, 3, 4, 5], [7, 3, 4, 5]),
            dim=0,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 9, 5], [2, 3, 4, 5], [2, 3, 1, 5]),
            dim=2,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4, 5], [2, 3, 4, 3], [2, 3, 4, 5]),
            dim=3,
        )

        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([12, 3, 4, 5], [3, 3, 4, 5], [7, 3, 4, 5]),
            dim=0,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]),
            dim=1,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 9, 5], [2, 3, 4, 5], [2, 3, 1, 5]),
            dim=2,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4, 5], [2, 3, 4, 3], [2, 3, 4, 5]),
            dim=3,
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4, 5], [2, 3, 4, 3], [2, 3, 4, 5]),
            dim=-1,
        )
        # special cases where inputs contain a number of zero-size tensor
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([3, 0], [3, 0]), dim=0
        )
        self._run_concatenate(
            concatenate_op=ops.concatenate(), input_shapes=([3, 0], [3, 1]), dim=1
        )

    def test_masked_cat(self):
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2], [2]),
            input_masks=[True, False],
            dim=0,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3], [5, 3], [3, 3]),
            input_masks=[False, True, True],
            dim=0,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 11, 4], [2, 5, 4], [2, 2, 4]),
            input_masks=[True, False, True],
            dim=1,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([1, 1, 1], [1, 1, 2], [1, 1, 4]),
            input_masks=[False, True, False],
            dim=2,
        )

    @parameterized.expand(("float16", "float32", "bfloat16"))
    def test_floats(self, dtype):
        if detect_target().name() != "cuda" and dtype != "float16":
            self.skipTest(
                f"{detect_target().name()} backend is not supported for {dtype} input type"
            )
        self._run_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([1, 3, 1], [2, 3, 1], [3, 3, 1]),
            input_type=dtype,
        )
        self._run_masked_concatenate(
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [2, 3, 8], [2, 3, 16]),
            input_masks=[False, True, False],
            dim=2,
            input_type=dtype,
        )
        self._run_batch_concatenate(
            batch_sizes=[3, 5, 9],
            concatenate_op=ops.concatenate(),
            input_shapes=([2, 3, 4], [2, 3, 2]),
            dim=3,
            input_type=dtype,
        )

    def _test_concatenate_shape(self, in_shapes, out_shape, dim):
        Xs = [
            Tensor(
                shape=in_shape,
                name=f"input_{idx}",
                is_input=True,
            )
            for idx, in_shape in enumerate(in_shapes)
        ]

        Y = ops.concatenate()(Xs, dim)

        y_shape = Y.shape()
        self.assertEqual(len(y_shape), len(out_shape))
        for y, o in zip(y_shape, out_shape):
            self.assertEqual(y, o)

    def test_concatenate_shape_var(self):
        var1 = IntVar(values=[1, 2], name="var1")
        var2 = IntVar(values=[3, 5], name="var2")
        var3 = IntVar(values=[7, 11], name="var3")
        sym1 = var1._attrs["symbolic_value"]
        sym2 = var2._attrs["symbolic_value"]
        sym3 = var3._attrs["symbolic_value"]

        in_shapes = [[var, 2, 3] for var in [var1, var2, var3]]
        ovar1 = IntVar(values=[11, 18], symbolic_value=sym1 + sym2 + sym3)
        self._test_concatenate_shape(in_shapes, [ovar1, 2, 3], 0)
        self._test_concatenate_shape(in_shapes, [ovar1, 2, 3], -3)

    def test_concatenate_shape_mix(self):
        var1 = IntVar(values=[1, 2], name="var1")
        var2 = IntVar(values=[3, 5], name="var2")
        imm1 = IntImm(17)
        imm2 = IntImm(19)
        sym1 = var1._attrs["symbolic_value"]
        sym2 = var2._attrs["symbolic_value"]

        in_shapes = [[var1, 2, 3], [imm1, 2, 3], [imm2, 2, 3], [var2, 2, 3]]
        ovar1 = IntVar(values=[40, 43], symbolic_value=sym1 + sym2 + 17 + 19)
        self._test_concatenate_shape(in_shapes, [ovar1, 2, 3], 0)

    def test_concatenate_shape_compatible(self):
        var1 = IntVar(values=[1, 2])
        sym1 = var1._attrs["symbolic_value"]

        in_shapes = [[var1, 2, 3], [var1, 2, 3]]
        self._test_concatenate_shape(in_shapes, [var1, 2, 6], -1)

        dup_var1 = IntVar(values=[1, 2], symbolic_value=sym1)
        in_shapes = [[var1, 2, 3], [dup_var1, 2, 3]]
        self._test_concatenate_shape(in_shapes, [var1, 2, 6], -1)

        var2 = IntVar(values=[1, 2])
        with self.assertRaises(RuntimeError):
            in_shapes = [[var1, 2, 3], [var2, 2, 3]]
            self._test_concatenate_shape(in_shapes, [var1, 2, 6], -1)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
