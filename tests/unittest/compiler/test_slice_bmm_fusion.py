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
from aitemplate.compiler.base import IntImm
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class SliceBMMFusionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(SliceBMMFusionTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _bmm_parameters(self, bmm_op_name, batch_sizes, M, N, K):
        """
        Return a dict of parameters used for constructing bmm ops
        """
        B_dim = shape_utils.gen_int_var_min_max(batch_sizes, "batch_size")
        M_dim = shape_utils.gen_int_var_min_max(M) if isinstance(M, list) else IntImm(M)
        N_dim = shape_utils.gen_int_var_min_max(N) if isinstance(N, list) else IntImm(N)
        K_dim = shape_utils.gen_int_var_min_max(K) if isinstance(K, list) else IntImm(K)
        a_shape = {
            "r": [B_dim, M_dim, K_dim],
            "c": [B_dim, K_dim, M_dim],
        }
        b_shape = {
            "r": [B_dim, K_dim, N_dim],
            "c": [B_dim, N_dim, K_dim],
        }
        c_shape = {
            "r": [B_dim, M_dim, N_dim],
            "c": [B_dim, N_dim, M_dim],
        }
        permute = {
            "r": None,
            "c": [0, 2, 1],
        }
        bmm_op_name = bmm_op_name[:7]
        a_layout = bmm_op_name[4]
        b_layout = bmm_op_name[5]
        c_layout = bmm_op_name[6]
        bmm_dict = {}
        bmm_dict["a_shape"] = a_shape.get(a_layout)
        bmm_dict["b_shape"] = b_shape.get(b_layout)
        bmm_dict["c_shape"] = c_shape.get(c_layout)
        bmm_dict["a_permute"] = permute.get(a_layout)
        bmm_dict["b_permute"] = permute.get(b_layout)
        bmm_dict["c_permute"] = permute.get(c_layout)
        return bmm_dict

    def _test_slice_bmm_xxx_fusion_a(
        self,
        bmm_op_fn,
        M,
        N,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        expected_num_tensors,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # bmm(slice_output, B)
        assert (
            len(slice_input_shape) == 3
        ), f"expected {slice_input_shape=} to have a rank of 3"
        Batch = slice_input_shape[0]
        batch_sizes = [1, Batch]
        bmm_op = bmm_op_fn()
        bmm_params = self._bmm_parameters(bmm_op._attrs["op"], batch_sizes, M, N, K)
        a_shape = bmm_params["a_shape"]

        slice_input_tensor_shape = [a_shape[0]] + [
            shape_utils.gen_int_var_min_max(d) if isinstance(d, list) else IntImm(d)
            for d in slice_input_shape[1:]
        ]
        X = Tensor(
            shape=slice_input_tensor_shape,
            dtype=dtype,
            name="x",
            is_input=True,
        )
        slice_op = ops.dynamic_slice()
        A = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        assert shape_utils.is_same_shape(
            a_shape, A.shape()
        ), f"expected {a_shape=} and {A.shape()=} are the same shape"
        b_shape = bmm_params["b_shape"]
        B = Tensor(
            shape=b_shape,
            dtype=dtype,
            name="b",
            is_input=True,
        )
        input_tensors = [A, B]
        c_shape = bmm_params["c_shape"]
        has_add = "_add" in bmm_op._attrs["op"]
        if has_add:
            D = Tensor(
                shape=c_shape,
                dtype=dtype,
                name="d",
                is_input=True,
            )
            input_tensors.append(D)
        Y = bmm_op(*input_tensors)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), expected_num_tensors)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)

        dynamic_dim = [d for d in slice_input_shape[1:] if isinstance(d, list)]
        assert (
            len(dynamic_dim) == 0 or len(dynamic_dim) == 1
        ), f"expected at most one dynamic dim besides batch dim in {slice_input_shape=}"
        if len(dynamic_dim) == 1:
            assert len(dynamic_dim[0]) == len(
                batch_sizes
            ), f"expected {dynamic_dim[0]} and {batch_sizes=} have the same rank"
        for idx, batch in enumerate(batch_sizes):
            input_shape_pt = [batch] + [
                d[idx] if isinstance(d, list) else d for d in slice_input_shape[1:]
            ]
            x_pt = get_random_torch_tensor(input_shape_pt, dtype)
            b_pt = get_random_torch_tensor(
                [batch, b_shape[1].value(), b_shape[2].value()], dtype
            )
            slice_indices = [
                slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
            ]
            a_pt = x_pt[slice_indices]

            a_permute = bmm_params["a_permute"]
            bmm_a_pt = a_pt
            if a_permute is not None:
                bmm_a_pt = a_pt.permute(a_permute)
            b_permute = bmm_params["b_permute"]
            bmm_b_pt = b_pt
            if b_permute is not None:
                bmm_b_pt = b_pt.permute(b_permute)
            y_pt = torch.bmm(bmm_a_pt, bmm_b_pt)
            c_permute = bmm_params["c_permute"]
            bmm_y_pt = y_pt
            if c_permute is not None:
                bmm_y_pt = y_pt.permute(c_permute)

            inputs = {"x": x_pt, "b": b_pt}
            if has_add:
                d_pt = get_random_torch_tensor(
                    [batch, c_shape[-2].value(), c_shape[-1].value()], dtype
                )
                inputs["d"] = d_pt
                bmm_y_pt = bmm_y_pt + d_pt
            y = get_torch_empty_tensor(bmm_y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(bmm_y_pt, y, atol=0.01, rtol=0.01)

    def test_slice_bmm_rcr_fusion_a(self):
        # non-fusible due to the odd K
        slice_start_indices = [0, 1, 0]
        slice_end_indices = [None, 7, None]
        K = 5
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rcr,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=4,
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=7,
            expected_num_ops=3,
            test_name="slice_bmm_rcr_fusion_a",
        )

        slice_start_indices = [0, 0, 0]
        slice_end_indices = [None, 4, None]
        K = 8
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rcr,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=4,
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=3,
            expected_num_ops=1,
            test_name="slice_bmm_rcr_fusion_a",
        )

        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 7, None]
        K = 2
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rcr_add,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=4,
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_rcr_fusion_a",
        )

    def test_slice_bmm_rrr_fusion_a(self):
        # non-fusible
        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 6, None]
        K = 7
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rrr_add,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=4,
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=8,
            expected_num_ops=3,
            test_name="slice_bmm_rrr_fusion_a",
        )

        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 6, None]
        K = 4
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rrr_add,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=8,
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_rrr_fusion_a",
        )

    def test_slice_bmm_rrc_fusion_a(self):
        # non-fusible due to dynamic dimension
        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 6, None]
        K = 2
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rrc_add,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=8,
            K=K,
            slice_input_shape=(2, [10, 20], K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=5,
            expected_num_ops=2,
            test_name="slice_bmm_rrc_fusion_dynamic_a",
        )

        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 6, None]
        K = 2
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rrc_add,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=8,
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_rrc_fusion_a",
        )

    def test_slice_bmm_crr_fusion_a(self):
        # non-fusible
        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 6, None]
        M = 3
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_crr_add,
            M=M,
            N=6,
            K=(slice_end_indices[1] - slice_start_indices[1]),
            slice_input_shape=(2, 10, M),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=9,
            expected_num_ops=4,
            test_name="slice_bmm_crr_fusion_a",
        )

        slice_start_indices = [0, 3, 0]
        slice_end_indices = [None, 6, None]
        M = 8
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_crr_add,
            M=M,
            N=6,
            K=(slice_end_indices[1] - slice_start_indices[1]),
            slice_input_shape=(2, 10, M),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_crr_fusion_a",
        )

    def test_slice_bmm_rcc_fusion_a(self):
        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 7, None]
        K = 8
        self._test_slice_bmm_xxx_fusion_a(
            bmm_op_fn=ops.bmm_rcc,
            M=(slice_end_indices[1] - slice_start_indices[1]),
            N=4,
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=3,
            expected_num_ops=1,
            test_name="slice_bmm_rcc_fusion_a",
        )

    def _test_slice_bmm_xxx_fusion_b(
        self,
        bmm_op_fn,
        M,
        N,
        K,
        slice_input_shape,
        slice_start_indices,
        slice_end_indices,
        expected_num_tensors,
        expected_num_ops,
        test_name,
        dtype="float16",
    ):
        # bmm(A, slice_output)
        assert (
            len(slice_input_shape) == 3
        ), f"expected {slice_input_shape=} to have a rank of 3"
        Batch = slice_input_shape[0]
        batch_sizes = [1, Batch]
        bmm_op = bmm_op_fn()
        bmm_params = self._bmm_parameters(bmm_op._attrs["op"], batch_sizes, M, N, K)
        b_shape = bmm_params["b_shape"]

        slice_input_tensor_shape = [b_shape[0]] + [
            shape_utils.gen_int_var_min_max(d) if isinstance(d, list) else IntImm(d)
            for d in slice_input_shape[1:]
        ]
        X = Tensor(
            shape=slice_input_tensor_shape,
            dtype=dtype,
            name="x",
            is_input=True,
        )
        slice_op = ops.dynamic_slice()
        B = slice_op(
            X, start_indices=slice_start_indices, end_indices=slice_end_indices
        )
        a_shape = bmm_params["a_shape"]
        A = Tensor(
            shape=a_shape,
            dtype=dtype,
            name="a",
            is_input=True,
        )
        assert shape_utils.is_same_shape(
            b_shape, B.shape()
        ), f"expected {b_shape=} and {B.shape()=} are the same shape"
        input_tensors = [A, B]
        c_shape = bmm_params["c_shape"]
        has_add = "_add" in bmm_op._attrs["op"]
        if has_add:
            D = Tensor(
                shape=c_shape,
                dtype=dtype,
                name="d",
                is_input=True,
            )
            input_tensors.append(D)
        Y = bmm_op(*input_tensors)
        Y._attrs["name"] = "output"
        Y._attrs["is_output"] = True

        target = detect_target()
        dll_name = "test_{}.so".format(self.test_count)
        module = compile_model(Y, target, "./tmp", test_name, dll_name=dll_name)

        # Verify the generated graph.
        sorted_graph = module.debug_sorted_graph
        self.assertEqual(len(sorted_graph), expected_num_tensors)
        sorted_ops = graph_utils.get_sorted_ops(sorted_graph)
        self.assertEqual(len(sorted_ops), expected_num_ops)

        dynamic_dim = [d for d in slice_input_shape[1:] if isinstance(d, list)]
        assert (
            len(dynamic_dim) == 0 or len(dynamic_dim) == 1
        ), f"expected at most one dynamic dim besides batch dim in {slice_input_shape=}"
        if len(dynamic_dim) == 1:
            assert len(dynamic_dim[0]) == len(
                batch_sizes
            ), f"expected {dynamic_dim[0]} and {batch_sizes=} have the same rank"
        for idx, batch in enumerate(batch_sizes):
            input_shape_pt = [batch] + [
                d[idx] if isinstance(d, list) else d for d in slice_input_shape[1:]
            ]
            x_pt = get_random_torch_tensor(input_shape_pt, dtype)
            a_pt = get_random_torch_tensor(
                [batch, a_shape[1].value(), a_shape[2].value()], dtype
            )
            slice_indices = [
                slice(i, j) for i, j in zip(slice_start_indices, slice_end_indices)
            ]
            b_pt = x_pt[slice_indices]

            a_permute = bmm_params["a_permute"]
            bmm_a_pt = a_pt
            if a_permute is not None:
                bmm_a_pt = a_pt.permute(a_permute)
            b_permute = bmm_params["b_permute"]
            bmm_b_pt = b_pt
            if b_permute is not None:
                bmm_b_pt = b_pt.permute(b_permute)
            y_pt = torch.bmm(bmm_a_pt, bmm_b_pt)
            c_permute = bmm_params["c_permute"]
            bmm_y_pt = y_pt
            if c_permute is not None:
                bmm_y_pt = y_pt.permute(c_permute)

            inputs = {"x": x_pt, "a": a_pt}
            if has_add:
                d_pt = get_random_torch_tensor(
                    [batch, c_shape[-2].value(), c_shape[-1].value()], dtype
                )
                inputs["d"] = d_pt
                bmm_y_pt = bmm_y_pt + d_pt
            y = get_torch_empty_tensor(bmm_y_pt.size(), dtype)
            module.run_with_tensors(inputs, [y])
            torch.testing.assert_close(bmm_y_pt, y, atol=0.01, rtol=0.01)

    def test_slice_bmm_rrc_fusion_b(self):
        slice_start_indices = [0, 2, 0]
        slice_end_indices = [None, 6, None]
        N = 2
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_rrc_add,
            M=8,
            N=N,
            K=(slice_end_indices[1] - slice_start_indices[1]),
            slice_input_shape=(2, 10, N),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_rrc_fusion_b",
        )

    def test_slice_bmm_crc_fusion_b(self):
        slice_start_indices = [0, 1, 0]
        slice_end_indices = [None, 6, None]
        N = 4
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_crc_add,
            M=8,
            N=N,
            K=(slice_end_indices[1] - slice_start_indices[1]),
            slice_input_shape=(2, 10, N),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_crc_fusion_b",
        )

    def test_slice_bmm_ccr_fusion_b(self):
        # non-fusible
        slice_start_indices = [0, 0, 2]
        slice_end_indices = [None, None, 6]
        N = 8
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_ccr_add,
            M=6,
            N=N,
            K=(slice_end_indices[-1] - slice_start_indices[-1]),
            slice_input_shape=(2, N, 7),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=5,
            expected_num_ops=2,
            test_name="slice_bmm_ccr_fusion_b",
        )

        slice_start_indices = [0, 1, 0]
        slice_end_indices = [None, 6, None]
        K = 4
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_ccr_add,
            M=8,
            N=(slice_end_indices[1] - slice_start_indices[1]),
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_ccr_fusion_b",
        )

    def test_slice_bmm_ccc_fusion_b(self):
        # non-fusible
        slice_start_indices = [0, 1, 0]
        slice_end_indices = [None, 6, None]
        K = 4
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_ccc_add,
            M=5,
            N=(slice_end_indices[1] - slice_start_indices[1]),
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=9,
            expected_num_ops=4,
            test_name="slice_bmm_ccc_fusion_b",
        )

        slice_start_indices = [0, 1, 0]
        slice_end_indices = [None, 6, None]
        K = 4
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_ccc_add,
            M=8,
            N=(slice_end_indices[1] - slice_start_indices[1]),
            K=K,
            slice_input_shape=(2, 10, K),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=4,
            expected_num_ops=1,
            test_name="slice_bmm_ccc_fusion_b",
        )

    def test_slice_bmm_rrr_fusion_b(self):
        # non-fusible
        slice_start_indices = [0, 0, 0]
        slice_end_indices = [None, None, 4]
        K = 8
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_rrr_add,
            M=9,
            N=(slice_end_indices[-1] - slice_start_indices[-1]),
            K=K,
            slice_input_shape=(2, K, 7),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=5,
            expected_num_ops=2,
            test_name="slice_bmm_rrr_fusion_b",
        )

        # non-fusible due to dynamic cim
        slice_start_indices = [0, 0, 0]
        slice_end_indices = [None, None, 4]
        K = 8
        self._test_slice_bmm_xxx_fusion_b(
            bmm_op_fn=ops.bmm_rrr_add,
            M=4,
            N=(slice_end_indices[-1] - slice_start_indices[-1]),
            K=K,
            slice_input_shape=(2, K, [10, 20]),
            slice_start_indices=slice_start_indices,
            slice_end_indices=slice_end_indices,
            expected_num_tensors=5,
            expected_num_ops=2,
            test_name="slice_bmm_rrr_fusion_dynamic_b",
        )


if __name__ == "__main__":
    unittest.main()
