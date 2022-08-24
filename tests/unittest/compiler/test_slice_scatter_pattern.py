# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import unittest

import numpy as np
import torch

from aitemplate.compiler import ops, transform
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target, gen_execution_module
from aitemplate.utils import graph_utils


class SliceScatterPatternTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceScatterPatternTestCase, self).__init__(*args, **kwargs)

    def _make_slice_ops(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
        batch_sizes=None,
        input_type="float16",
    ):
        Ys = []
        for idx, (input_shape, start_indices, end_indices) in enumerate(
            zip(input_shapes, input_start_indices, input_end_indices)
        ):
            slice_op = ops.dynamic_slice()
            X_name = "input_{}".format(idx)
            if batch_sizes is None:
                X = Tensor(
                    shape=input_shape, dtype=input_type, name=X_name, is_input=True
                )
            else:
                X = Tensor(
                    shape=[
                        IntVar(values=batch_sizes, name="input_batch_{}".format(idx)),
                        *input_shape,
                    ],
                    dtype=input_type,
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
        batch_sizes=None,
    ):
        Ys = self._make_slice_ops(
            input_shapes, input_start_indices, input_end_indices, dim, batch_sizes
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
        batch_sizes=None,
    ):
        graph = self._make_test_graph(
            input_shapes, input_start_indices, input_end_indices, dim, batch_sizes
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

    def _e2e_test(self, input_shapes, input_start_indices, input_end_indices, dim):
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
            X_pt = torch.randn(input_shape).cuda().half()
            Xs_pt.append(X_pt)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            Ys_pt.append(Y_pt)
        Y_pt = torch.cat(Ys_pt, dim)

        Y = self._make_test_graph(
            input_shapes, input_start_indices, input_end_indices, dim
        )
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        logging.info(
            "AITemplate output_0 shape: {}, pt shape: {}".format(y_shape, Y_pt.size())
        )
        np.testing.assert_equal(y_shape, Y_pt.size())

        module = gen_execution_module(Y, target, "./tmp", "slice_scatter")

        input_name_to_index = module.GetInputNameToIndexMap()
        inputs = [0 for i in range(len(Xs_pt))]
        for i, X_pt in enumerate(Xs_pt):
            inputs[input_name_to_index[f"input_{i}"]] = X_pt
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _e2e_batch_test(
        self, input_shapes, input_start_indices, input_end_indices, dim, batch_sizes
    ):
        logging.info(
            "e2e batch test with batch_sizes {}, input_shapes{}, "
            "start_indices {}, end_indices {}".format(
                batch_sizes, input_shapes, input_start_indices, input_end_indices
            )
        )

        target = detect_target()

        Y = self._make_test_graph(
            input_shapes, input_start_indices, input_end_indices, dim, batch_sizes
        )
        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]

        module = gen_execution_module(Y, target, "./tmp", "slice_scatter")

        for batch in batch_sizes:
            logging.info("checking batch: {}".format(batch))

            Ys_pt = []
            Xs_pt = []
            for input_shape, start_indices, end_indices in zip(
                input_shapes, input_start_indices, input_end_indices
            ):
                X_pt = torch.randn([batch, *input_shape]).cuda().half()
                Xs_pt.append(X_pt)
                slice_indices = [
                    slice(i, j) for i, j in zip(start_indices, end_indices)
                ]
                Y_pt = X_pt[slice_indices]
                Ys_pt.append(Y_pt)
            Y_pt = torch.cat(Ys_pt, dim)
            input_name_to_index = module.GetInputNameToIndexMap()
            inputs = [0 for i in range(len(Xs_pt))]
            for i, X_pt in enumerate(Xs_pt):
                inputs[input_name_to_index[f"input_{i}"]] = X_pt
            y = torch.empty(y_shape).cuda().half()
            module.RunWithTensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def _run_one_test(
        self, *, input_shapes, input_start_indices, input_end_indices, dim
    ):
        self._graph_transformation_test(
            input_shapes, input_start_indices, input_end_indices, dim
        )
        self._e2e_test(input_shapes, input_start_indices, input_end_indices, dim)

    def _run_one_batch_test(
        self, *, batch_sizes, input_shapes, input_start_indices, input_end_indices, dim
    ):
        self._graph_transformation_test(
            input_shapes, input_start_indices, input_end_indices, dim, batch_sizes
        )
        self._e2e_batch_test(
            input_shapes, input_start_indices, input_end_indices, dim, batch_sizes
        )

    def test_slice_scatter(self):
        self._run_one_test(
            input_shapes=[[2], [3]],
            input_start_indices=[[1], [2]],
            input_end_indices=[[2], [-1]],
            dim=0,
        )
        self._run_one_test(
            input_shapes=[[2, 3, 4], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [2, 5, 3]],
            input_end_indices=[[2, 2, 4], [-1, 6, 7]],
            dim=0,
        )
        self._run_one_test(
            input_shapes=[[2, 3, 4], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [0, 1, 3]],
            input_end_indices=[[2, 2, 4], [2, 7, 7]],
            dim=1,
        )
        self._run_one_test(
            input_shapes=[[2, 3, 4], [3, 7, 10]],
            input_start_indices=[[0, 0, -3], [0, 2, 1]],
            input_end_indices=[[2, 3, 4], [2, 5, -1]],
            dim=2,
        )

    def test_batch_slice_scatter(self):
        self._run_one_batch_test(
            batch_sizes=[1024, 4, 128],
            input_shapes=[[3], [2]],
            input_start_indices=[[1, 1], [0, 0]],
            input_end_indices=[[2, 3], [1, 2]],
            dim=0,
        )
        self._run_one_batch_test(
            batch_sizes=[4, 3, 7],
            input_shapes=[[2], [3]],
            input_start_indices=[[1, 0], [0, 0]],
            input_end_indices=[[2, 1], [1, -1]],
            dim=1,
        )

    def _make_test_graph_multi_dsts(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
    ):
        Ys = self._make_slice_ops(
            input_shapes,
            input_start_indices,
            input_end_indices,
            dim,
        )
        input_type = "float16"
        # make the first input tensor have multiple uses
        slice_op_0 = list(Ys[0].src_ops())[0]
        X0 = slice_op_0._attrs["inputs"][0]
        X0_shape = [d._attrs["values"][0] for d in X0._attrs["shape"]]
        num_slice_inputs = len(input_shapes)
        X1_name = f"input_{num_slice_inputs}"
        X1 = Tensor(shape=X0_shape, dtype=input_type, name=X1_name, is_input=True)
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
        self, *, input_shapes, input_start_indices, input_end_indices, dim
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
            X_pt = torch.randn(input_shape).cuda().half()
            Xs_pt.append(X_pt)
            slice_indices = [slice(i, j) for i, j in zip(start_indices, end_indices)]
            Y_pt = X_pt[slice_indices]
            Ys_pt.append(Y_pt)
        Y0_pt = torch.cat(Ys_pt, dim)

        input0_shape = Xs_pt[0].size()
        other_X_pt = torch.randn(input0_shape).cuda().half()
        Xs_pt.append(other_X_pt)
        Y1_pt = Xs_pt[0] + other_X_pt

        Y0, Y1 = self._make_test_graph_multi_dsts(
            input_shapes, input_start_indices, input_end_indices, dim
        )

        y0_shape = [var._attrs["values"][0] for var in Y0._attrs["shape"]]
        y1_shape = [var._attrs["values"][0] for var in Y1._attrs["shape"]]
        np.testing.assert_equal(y0_shape, Y0_pt.size())
        np.testing.assert_equal(y1_shape, Y1_pt.size())

        test_name = "slice_scatter_multi_dsts"
        module = gen_execution_module((Y0, Y1), target, "./tmp", test_name)
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

        input_name_to_index = module.GetInputNameToIndexMap()
        inputs = [0 for i in range(len(Xs_pt))]
        for i, X_pt in enumerate(Xs_pt):
            inputs[input_name_to_index[f"input_{i}"]] = X_pt
        y0 = torch.empty(y0_shape).cuda().half()
        y1 = torch.empty(y1_shape).cuda().half()
        module.RunWithTensors(inputs, [y0, y1])
        self.assertTrue(torch.allclose(Y0_pt, y0, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(Y1_pt, y1, atol=1e-2, rtol=1e-2))

    def test_slice_scatter_multi_dsts(self):
        self._test_slice_scatter_multi_dsts(
            input_shapes=[[2], [3]],
            input_start_indices=[[1], [2]],
            input_end_indices=[[2], [-1]],
            dim=0,
        )
        self._test_slice_scatter_multi_dsts(
            input_shapes=[[2, 3, 4], [3, 7, 10]],
            input_start_indices=[[0, 0, -3], [0, 2, 1]],
            input_end_indices=[[2, 3, 4], [2, 5, -1]],
            dim=2,
        )

    def _make_test_graph_multi_dsts_2(
        self,
        input_shapes,
        input_start_indices,
        input_end_indices,
        dim,
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
        self, *, input_shapes, input_start_indices, input_end_indices, dim
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
            X_pt = torch.randn(input_shape).cuda().half()
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
            input_shapes, input_start_indices, input_end_indices, dim
        )

        y_shape = [var._attrs["values"][0] for var in Y._attrs["shape"]]
        np.testing.assert_equal(y_shape, Y_pt.size())

        test_name = "slice_scatter_multi_dsts_2"
        module = gen_execution_module(Y, target, "./tmp", test_name)
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

        input_name_to_index = module.GetInputNameToIndexMap()
        inputs = [0 for i in range(len(Xs_pt))]
        for i, X_pt in enumerate(Xs_pt):
            inputs[input_name_to_index[f"input_{i}"]] = X_pt
        y = torch.empty(y_shape).cuda().half()
        module.RunWithTensors(inputs, [y])
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_slice_scatter_multi_dsts_2(self):
        self._test_slice_scatter_multi_dsts_2(
            input_shapes=[[2, 3, 4], [3, 7, 10]],
            input_start_indices=[[0, 1, 0], [0, 1, 3]],
            input_end_indices=[[2, 2, 4], [2, 7, 7]],
            dim=1,
        )
        self._test_slice_scatter_multi_dsts_2(
            input_shapes=[[2, 3, 4], [3, 7, 10]],
            input_start_indices=[[0, 0, -3], [0, 2, 1]],
            input_end_indices=[[2, 3, 4], [2, 5, -1]],
            dim=2,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
