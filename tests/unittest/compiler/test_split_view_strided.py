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
import itertools
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target, test_utils
from aitemplate.testing.test_utils import (
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils import graph_utils, shape_utils


class SplitViewStridedOpTestCase(unittest.TestCase):
    def _test_split_view_bmm_rcr(
        self,
        bmm_rcr_op,
        Bs,
        Ms,
        input_A_shape,
        input_B_shape,
        split_size_or_sections,
        split_dim,
        reshape_A,
        reshape_B,
        expected_num_tensors,
        expected_num_ops,
        testname,
        dtype="float16",
    ):
        T_A = Tensor(
            shape=input_A_shape,
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        T_B = Tensor(
            shape=input_B_shape,
            dtype=dtype,
            name="input1",
            is_input=True,
        )
        Xs = ops.split()(T_A, split_size_or_sections, split_dim)
        Ys = ops.split()(T_B, split_size_or_sections, split_dim)
        self.assertEqual(len(Xs), len(Ys))

        n = len(Xs)
        Cs = []
        for i in range(n):
            X = ops.reshape()(Xs[i], reshape_A)
            Y = ops.reshape()(Ys[i], reshape_B)
            C = bmm_rcr_op()(X, Y)
            C._attrs["name"] = f"output_{i}"
            C._attrs["is_output"] = True
            Cs.append(C)

        # Gen module.
        target = detect_target()
        module = compile_model(Cs, target, "./tmp", testname)
        graph = module.debug_sorted_graph
        self.assertEqual(len(graph), expected_num_tensors)
        self.assertEqual(len(graph_utils.get_sorted_ops(graph)), expected_num_ops)

        for B, M in itertools.product(Bs, Ms):
            dim_to_value_dict = {
                "batch_size": B,
                "emb_pool_size": M,
            }
            a = get_random_torch_tensor(
                test_utils.get_shape(T_A._attrs["shape"], dim_to_value_dict),
                dtype,
            )
            b = get_random_torch_tensor(
                test_utils.get_shape(T_B._attrs["shape"], dim_to_value_dict),
                dtype,
            )
            xs = a.split(split_size_or_sections, split_dim)
            ys = b.split(split_size_or_sections, split_dim)
            cs = []
            for i in range(n):
                x = torch.reshape(xs[i], reshape_A)
                y = torch.reshape(ys[i], reshape_B)
                c = torch.bmm(x, y.permute(0, 2, 1))
                cs.append(c)

            ys = [get_torch_empty_tensor(y_pt.size(), dtype) for y_pt in cs]
            module.run_with_tensors({"input0": a, "input1": b}, ys)

            for y, y_pt in zip(ys, cs):
                self.assertTrue(
                    torch.allclose(y, y_pt, atol=1e-2, rtol=1e-2),
                    f"y: {y}\ny_pt: {y_pt}",
                )

    def test_split_view_bmm_rcr_fusion(self):
        b_dim = shape_utils.gen_int_var([1, 1024], "batch_size")
        m_dim = shape_utils.gen_int_var([100, 200], "emb_pool_size")

        # bmm_rcr dynamic M fusible
        self._test_split_view_bmm_rcr(
            ops.bmm_rcr,
            Bs=[1],
            Ms=[100, 105, 160],
            input_A_shape=[1, m_dim, 10, 2],
            input_B_shape=[1, 6, 10, 2],
            split_size_or_sections=10,
            split_dim=2,
            reshape_A=[1, -1, 20],
            reshape_B=[1, 6, 20],
            expected_num_tensors=3,
            expected_num_ops=1,
            testname="test_split_bmm_rcr_dynamic_m_fusible",
        )

        # bmm_rcr_n1 dynamic B fusible
        self._test_split_view_bmm_rcr(
            ops.bmm_rcr_n1,
            Bs=[2, 4, 5, 10],
            Ms=[100],
            input_A_shape=[b_dim, 100, 10, 4],
            input_B_shape=[b_dim, 1, 10, 4],
            split_size_or_sections=2,
            split_dim=2,
            reshape_A=[-1, 100, 8],
            reshape_B=[-1, 1, 8],
            expected_num_tensors=7,
            expected_num_ops=5,
            testname="test_split_bmm_rcr_n1_dynamic_b_fusible",
        )

        # bmm_rcr dynamic M unfusible
        self._test_split_view_bmm_rcr(
            ops.bmm_rcr,
            Bs=[2],
            Ms=[100, 200],
            input_A_shape=[2, m_dim, 10, 10],
            input_B_shape=[2, 6, 10, 10],
            split_size_or_sections=5,
            split_dim=3,
            reshape_A=[2, -1, 50],
            reshape_B=[2, -1, 50],
            expected_num_tensors=8,
            expected_num_ops=4,
            testname="test_split_bmm_rcr_dynamic_m_non_fusible",
        )

        # bmm_rcr dynamic M, B unfusible
        self._test_split_view_bmm_rcr(
            ops.bmm_rcr,
            Bs=[2, 4, 5, 10],
            Ms=[100, 200],
            input_A_shape=[b_dim, m_dim, 10, 8],
            input_B_shape=[b_dim, m_dim, 10, 8],
            split_size_or_sections=2,
            split_dim=2,
            reshape_A=[-1, 10, 16],
            reshape_B=[-1, 10, 16],
            expected_num_tensors=27,
            expected_num_ops=17,
            testname="test_split_bmm_rcr_dynamic_bm_non_fusible",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_split_view_bmm_rcr_fusion_fp32_sm80(self):
        b_dim = shape_utils.gen_int_var([1, 1024], "batch_size")
        m_dim = shape_utils.gen_int_var([100, 200], "emb_pool_size")

        # bmm_rcr dynamic M fusible
        self._test_split_view_bmm_rcr(
            ops.bmm_rcr,
            Bs=[1],
            Ms=[100, 105, 160],
            input_A_shape=[1, m_dim, 10, 2],
            input_B_shape=[1, 6, 10, 2],
            split_size_or_sections=10,
            split_dim=2,
            reshape_A=[1, -1, 20],
            reshape_B=[1, 6, 20],
            expected_num_tensors=3,
            expected_num_ops=1,
            testname="test_split_bmm_rcr_dynamic_m_fusible_float",
            dtype="float",
        )
        # bmm_rcr dynamic M, B unfusible
        self._test_split_view_bmm_rcr(
            ops.bmm_rcr,
            Bs=[2, 4, 5, 10],
            Ms=[100, 200],
            input_A_shape=[b_dim, m_dim, 10, 8],
            input_B_shape=[b_dim, m_dim, 10, 8],
            split_size_or_sections=2,
            split_dim=2,
            reshape_A=[-1, 10, 16],
            reshape_B=[-1, 10, 16],
            expected_num_tensors=27,
            expected_num_ops=17,
            testname="test_split_bmm_rcr_dynamic_bm_non_fusible_float",
            dtype="float",
        )


filter_test_cases_by_test_env(SplitViewStridedOpTestCase)

if __name__ == "__main__":
    unittest.main()
