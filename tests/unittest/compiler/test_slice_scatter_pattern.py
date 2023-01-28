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

from aitemplate.compiler import compile_model, ops, transform
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils


class SliceScatterPatternTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceScatterPatternTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _make_slice_ops(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype,
        batch_sizes=None,
    ):
        Ys = []
        for idx, (input_shape, start_indices, end_indices) in enumerate(
            zip(input_shapes, input_start_indices, input_end_indices)
        ):
            slice_op = ops.dynamic_slice()
            X_name = "input_{}".format(idx)
            if batch_sizes is None:
                X = Tensor(shape=input_shape, dtype=dtype, name=X_name, is_input=True)
            else:
                X = Tensor(
                    shape=[
                        IntVar(values=batch_sizes, name="input_batch_{}".format(idx)),
                        *input_shape,
                    ],
                    dtype=dtype,
                    name=X_name,
                    is_input=True,
                )
            Y = slice_op(X, start_indices=start_indices, end_indices=end_indices)
            Ys.append(Y)
        return Ys

    def _make_test_graph(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype,
        batch_sizes=None,
    ):
        Ys = self._make_slice_ops(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
            dtype,
            batch_sizes,
        )
        concat_op = ops.concatenate()
        Y = concat_op(Ys, dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        return Y

    def _graph_transformation_test(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype,
        batch_sizes=None,
    ):
        graph = self._make_test_graph(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
            dtype,
            batch_sizes,
        )
        graph = transform.toposort(graph)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        orig_graph_size = len(graph)
        graph = transform.transform_strided_ops(graph)
        self.assertEqual(len(graph), orig_graph_size - len(input_shapes))

        Y = graph[-1]
        self.assertEqual(Y._attrs["name"], "output_0")
        self.assertTrue(Y._attrs["is_output"])
        self.assertNotEqual(len(Y.src_ops()), 0)
        fused_op = list(Y.src_ops())[0]
        self.assertEqual(fused_op._attrs["op"], "slice_scatter")
        for idx, x in enumerate(fused_op._attrs["inputs"]):
            self.assertEqual(x._attrs["name"], "input_{}".format(idx))

    def _e2e_test(
        self, input_shapes, input_start_indices, input_end_indices, dim, dtype
    ):
        logging.info(
            "e2e test with input_shapes {}, start_indices {}, end_indices {}".format(
                input_shapes, input_start_indices, input_end_indices
            )
        )

        target = detect_target()

        Ys_pt = []
        Xs_pt = []
        for input_shape, start_indices, end_indices in zip(
            input_shapes, input_start_indices, input_end_indices
        ):
            X_pt = get_random_torch_tensor(input_shape, dtype)
            Xs_pt.append(X_pt)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            Ys_pt.append(Y_pt)
        Y_pt = torch.cat(Ys_pt, dim)

        Y = self._make_test_graph(
            input_shapes, input_start_indices, input_end_indices, dim, dtype
        )
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info(
            "AITemplate output_0 shape: {}, pt shape: {}".format(y_shape, Y_pt.size())
        )
        np.testing.assert_equal(y_shape, Y_pt.size())

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", "slice_scatter_e2e", dll_name=dll_name
        )

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(Xs_pt))]
        for i, X_pt in enumerate(Xs_pt):
            inputs[input_name_to_index[f"input_{i}"]] = X_pt
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def _e2e_batch_test(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype,
        batch_sizes,
    ):
        logging.info(
            "e2e batch test with batch_sizes {}, input_shapes{}, "
            "start_indices {}, end_indices {}".format(
                batch_sizes, input_shapes, input_start_indices, input_end_indices
            )
        )

        target = detect_target()

        Y = self._make_test_graph(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
            dtype,
            batch_sizes,
        )
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]

        dll_name = f"test_{self.test_count}.so"
        module = compile_model(
            Y, target, "./tmp", "slice_scatter_e2d_batch", dll_name=dll_name
        )

        for batch in batch_sizes:
            logging.info("checking batch: {}".format(batch))

            Ys_pt = []
            Xs_pt = []
            for input_shape, start_indices, end_indices in zip(
                input_shapes, input_start_indices, input_end_indices
            ):
                X_pt = get_random_torch_tensor([batch, *input_shape], dtype)
                Xs_pt.append(X_pt)
                slice_indices = [
                    slice(i, j) for i, j in zip(start_indices, end_indices)
                ]
                Y_pt = X_pt[slice_indices]
                Ys_pt.append(Y_pt)
            Y_pt = torch.cat(Ys_pt, dim)
            input_name_to_index = module.get_input_name_to_index_map()
            inputs = [0 for i in range(len(Xs_pt))]
            for i, X_pt in enumerate(Xs_pt):
                inputs[input_name_to_index[f"input_{i}"]] = X_pt
            y = get_torch_empty_tensor(y_shape, dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
            self.test_count += 1

    def _run_one_test(
        self,
        *,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype="float16",
    ):
        self._graph_transformation_test(
            input_shapes, input_start_indices, input_end_indices, dim, dtype
        )
        self._e2e_test(input_shapes, input_start_indices, input_end_indices, dim, dtype)

    def _run_one_batch_test(
        self,
        *,
        batch_sizes,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype="float16",
    ):
        self._graph_transformation_test(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
            dtype,
            batch_sizes,
        )
        self._e2e_batch_test(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
            dtype,
            batch_sizes,
        )

    def test_slice_scatter(self):
        self._run_one_test(
            input_shapes=[[2, 3, 5], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [2, 5, 3]],
            input_end_indices=[[2, 2, 4], [-1, 6, 7]],
            dim=0,
        )
        self._run_one_test(
            input_shapes=[[2, 3, 5], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [0, 1, 3]],
            input_end_indices=[[2, 2, 4], [2, 7, 7]],
            dim=1,
        )
        self._run_one_test(
            input_shapes=[[3, 3, 4], [3, 7, 10]],
            input_start_indices=[[1, 0, -3], [0, 2, 1]],
            input_end_indices=[[3, 3, 4], [2, 5, -1]],
            dim=2,
        )

    def test_batch_slice_scatter(self):
        self._run_one_batch_test(
            batch_sizes=[1024, 4, 128],
            input_shapes=[[3], [3]],
            input_start_indices=[[1, 1], [0, 0]],
            input_end_indices=[[2, 3], [1, 2]],
            dim=0,
        )
        self._run_one_batch_test(
            batch_sizes=[4, 3, 7],
            input_shapes=[[2], [4]],
            input_start_indices=[[1, 0], [0, 1]],
            input_end_indices=[[2, 1], [1, -1]],
            dim=1,
        )

    def _make_test_graph_multi_dsts(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype,
    ):
        Ys = self._make_slice_ops(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
            dtype,
        )
        # make the first input tensor have multiple uses
        slice_op_0 = list(Ys[0].src_ops())[0]
        X0 = slice_op_0._attrs["inputs"][0]
        X0_shape = [d._attrs["values"][0] for d in X0._attrs["shape"]]
        num_slice_inputs = len(input_shapes)
        X1_name = f"input_{num_slice_inputs}"
        X1 = Tensor(shape=X0_shape, dtype=dtype, name=X1_name, is_input=True)
        concat_op = ops.concatenate()
        Y0 = concat_op(Ys, dim)
        Y0._attrs["name"] = "output_0"
        Y0._attrs["is_output"] = True

        add_op = ops.elementwise(FuncEnum.ADD)
        Y1 = add_op(X0, X1)
        Y1._attrs["name"] = "output_1"
        Y1._attrs["is_output"] = True

        return (Y0, Y1)

    def _test_slice_scatter_multi_dsts(
        self,
        *,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype="float16",
    ):
        """test cases where a tensor being sliced has multiple dsts"""

        logging.info(
            f"multi_dsts e2e test with input_shapes: {input_shapes}, "
            f"start_indices: {input_start_indices}, end_indices: {input_end_indices}"
        )
        target = detect_target()

        Ys_pt = []
        Xs_pt = []
        for input_shape, start_indices, end_indices in zip(
            input_shapes, input_start_indices, input_end_indices
        ):
            X_pt = get_random_torch_tensor(input_shape, dtype)
            Xs_pt.append(X_pt)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            Ys_pt.append(Y_pt)
        Y0_pt = torch.cat(Ys_pt, dim)

        input0_shape = Xs_pt[0].size()
        other_X_pt = get_random_torch_tensor(input0_shape, dtype)
        Xs_pt.append(other_X_pt)
        Y1_pt = Xs_pt[0] + other_X_pt

        Y0, Y1 = self._make_test_graph_multi_dsts(
            input_shapes, input_start_indices, input_end_indices, dim, dtype
        )

        y0_shape = [var._attrs["values"][0] for var in Y0._attrs["shape"]]
        y1_shape = [var._attrs["values"][0] for var in Y1._attrs["shape"]]
        np.testing.assert_equal(y0_shape, Y0_pt.size())
        np.testing.assert_equal(y1_shape, Y1_pt.size())

        test_name = "slice_scatter_multi_dsts"
        dll_name = f"test_{self.test_count}.so"
        module = compile_model((Y0, Y1), target, "./tmp", test_name, dll_name=dll_name)
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 2)
        if sorted_ops[0]._attrs["op"] == "fused_elementwise":
            slice_scatter_op = sorted_ops[1]
        else:
            slice_scatter_op = sorted_ops[0]
        self.assertEqual(slice_scatter_op._attrs["op"], "slice_scatter")
        for idx, x in enumerate(slice_scatter_op._attrs["inputs"]):
            self.assertEqual(x._attrs["name"], "input_{}".format(idx))

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(Xs_pt))]
        for i, X_pt in enumerate(Xs_pt):
            inputs[input_name_to_index[f"input_{i}"]] = X_pt
        y0 = get_torch_empty_tensor(y0_shape, dtype)
        y1 = get_torch_empty_tensor(y1_shape, dtype)
        module.run_with_tensors(inputs, [y0, y1])
        self.assertTrue(torch.allclose(Y0_pt, y0, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_scatter_multi_dsts(self):
        self._test_slice_scatter_multi_dsts(
            input_shapes=[[4, 3, 4], [3, 7, 10]],
            input_start_indices=[[1, 0, -3], [0, 2, 1]],
            input_end_indices=[[3, 3, 4], [2, 5, -1]],
            dim=2,
        )

    def _make_test_graph_multi_dsts_2(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype,
    ):
        """Make a graph where (1) a tensor is sliced twice and both slices are
        fed into the same concat op, and (2) another sliced output (i.e not
        the one from (1)) is fed into the same concat op twice.
        """

        Ys = self._make_slice_ops(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
            dtype,
        )
        slice_op_0 = list(Ys[0].src_ops())[0]
        X0 = slice_op_0._attrs["inputs"][0]
        # make one more slice op that takes the tensor input of the first slice op
        slice_op = ops.dynamic_slice()
        Y0 = slice_op(
            X0, start_indices=input_start_indices[0], end_indices=input_end_indices[0]
        )
        Ys.append(Y0)

        # The last sliced output is fed into concat twice
        Y_1 = Ys[-1]
        Ys.append(Y_1)

        concat_op = ops.concatenate()
        Y = concat_op(Ys, dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        return Y

    def _test_slice_scatter_multi_dsts_2(
        self,
        *,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        dtype="float16",
    ):
        logging.info(
            f"multi_dsts_2 e2e test with input_shapes: {input_shapes}, "
            f"start_indices: {input_start_indices}, end_indices: {input_end_indices}"
        )
        target = detect_target()

        Ys_pt = []
        Xs_pt = []
        for input_shape, start_indices, end_indices in zip(
            input_shapes, input_start_indices, input_end_indices
        ):
            X_pt = get_random_torch_tensor(input_shape, dtype)
            Xs_pt.append(X_pt)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            Ys_pt.append(Y_pt)
        X0_pt = Xs_pt[0]
        slice0_indices = [
            slice(i, j) for i, j in zip(input_start_indices[0], input_end_indices[0])
        ]
        Y0_pt = X0_pt[slice0_indices]
        Ys_pt.append(Y0_pt)

        Y1_pt = Ys_pt[-1]
        Ys_pt.append(Y1_pt)

        Y_pt = torch.cat(Ys_pt, dim)

        Y = self._make_test_graph_multi_dsts_2(
            input_shapes, input_start_indices, input_end_indices, dim, dtype
        )

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        np.testing.assert_equal(y_shape, Y_pt.size())

        test_name = "slice_scatter_multi_dsts_2"
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)
        debug_sorted_graph = module.debug_sorted_graph
        sorted_ops = graph_utils.get_sorted_ops(debug_sorted_graph)
        self.assertEqual(len(sorted_ops), 1)
        slice_scatter_op = sorted_ops[0]
        self.assertEqual(slice_scatter_op._attrs["op"], "slice_scatter")
        slice_scatter_inputs = slice_scatter_op._attrs["inputs"]
        for idx, x in enumerate(slice_scatter_inputs[:-2]):
            self.assertEqual(x._attrs["name"], "input_{}".format(idx))
        self.assertEqual(slice_scatter_inputs[-2]._attrs["name"], "input_0")
        self.assertEqual(slice_scatter_inputs[-1]._attrs["name"], "input_0")

        input_name_to_index = module.get_input_name_to_index_map()
        inputs = [0 for i in range(len(Xs_pt))]
        for i, X_pt in enumerate(Xs_pt):
            inputs[input_name_to_index[f"input_{i}"]] = X_pt
        y = get_torch_empty_tensor(y_shape, dtype)
        module.run_with_tensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))
        self.test_count += 1

    def test_slice_scatter_multi_dsts_2(self):
        self._test_slice_scatter_multi_dsts_2(
            input_shapes=[[2, 3, 5], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [0, 1, 3]],
            input_end_indices=[[2, 2, 4], [2, 7, 7]],
            dim=1,
        )
        self._test_slice_scatter_multi_dsts_2(
            input_shapes=[[2, 4, 4], [3, 7, 10]],
            input_start_indices=[[0, 0, -3], [0, 2, 1]],
            input_end_indices=[[2, 3, 4], [2, 5, -1]],
            dim=2,
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_slice_scatter_float(self):
        self._run_one_test(
            input_shapes=[[2, 3, 5], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [0, 1, 3]],
            input_end_indices=[[2, 2, 4], [2, 7, 7]],
            dim=1,
            dtype="float",
        )
        self._run_one_batch_test(
            batch_sizes=[1024, 4, 128],
            input_shapes=[[3], [3]],
            input_start_indices=[[1, 1], [0, 0]],
            input_end_indices=[[2, 3], [1, 2]],
            dim=0,
            dtype="float",
        )
        self._test_slice_scatter_multi_dsts(
            input_shapes=[[4, 3, 4], [3, 7, 10]],
            input_start_indices=[[1, 0, -3], [0, 2, 1]],
            input_end_indices=[[3, 3, 4], [2, 5, -1]],
            dim=2,
            dtype="float",
        )
        self._test_slice_scatter_multi_dsts_2(
            input_shapes=[[2, 3, 5], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [0, 1, 3]],
            input_end_indices=[[2, 2, 4], [2, 7, 7]],
            dim=1,
            dtype="float",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
