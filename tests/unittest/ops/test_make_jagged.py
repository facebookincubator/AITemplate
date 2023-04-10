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
from aitemplate.compiler.base import JaggedDim, JaggedIntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import IntImm, IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.jagged_utils import add_jagged_dense_ref
from aitemplate.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from aitemplate.utils.graph_utils import get_sorted_ops
from aitemplate.utils.torch_utils import string_to_torch_dtype


class MakeJaggedTestCase(unittest.TestCase):
    def _test_make_jagged(
        self,
        check_sequence_lengths=True,
        test_name="make_jagged",
    ):
        offsets1 = Tensor(
            shape=[
                IntVar(values=[1, 16]),
            ],
            name="off1",
            dtype="int32",
            is_input=True,
        )
        offsets2 = Tensor(
            shape=[
                IntVar(values=[1, 16]),
            ],
            name="off2",
            dtype="int32",
            is_input=True,
        )

        X = Tensor(
            shape=[
                IntVar(values=[1, 1024]),
                IntImm(value=128),
            ],
            name="X",
            dtype="float16",
            is_input=True,
        )
        W = Tensor(
            shape=[
                IntImm(value=128),
                IntImm(value=64),
            ],
            name="W",
            dtype="float16",
            is_input=True,
        )

        batch_dim = IntVar(values=[1, 128])
        jd0 = JaggedDim(min_value=0, max_value=2)
        jd1 = JaggedDim(min_value=0, max_value=3)
        Y = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=[jd0, jd1],
            check_sequence_lengths=check_sequence_lengths,
        )(X, [offsets1, offsets2])
        Z = ops.gemm_rrr()(Y, W)

        assert Y.is_jagged()
        assert Z.is_jagged()

        Y_dim_0 = Y._attrs["shape"][0]
        assert isinstance(Y_dim_0, JaggedIntVar)
        assert Y_dim_0.jagged_dims() == [jd0, jd1]
        assert jd0.offsets() == offsets1
        assert jd1.offsets() == offsets2

        Z_dim_0 = Z._attrs["shape"][0]
        assert Z_dim_0 == Y_dim_0

        Y._attrs["name"] = "Y"
        Y._attrs["is_output"] = True
        Z._attrs["name"] = "Z"
        Z._attrs["is_output"] = True

        model = compile_model([Y, Z], detect_target(), "./tmp", test_name)

        offsets1_pt = torch.tensor([0, 1, 3, 5], dtype=torch.int32).cuda()
        offsets2_pt = torch.tensor([0, 2, 4, 4, 7, 10], dtype=torch.int32).cuda()

        if not check_sequence_lengths:
            # extend seq lens beyond the JaggedDim bounds
            offsets1_pt[2] = 4
            offsets2_pt[4] = 9

        x_pt = get_random_torch_tensor([10, 128], "float16")
        w_pt = get_random_torch_tensor([128, 64], "float16")
        z_pt = torch.matmul(x_pt, w_pt)

        y = get_torch_empty_tensor([10, 128], "float16")
        z = get_torch_empty_tensor([10, 64], "float16")

        inputs = {"X": x_pt, "off1": offsets1_pt, "off2": offsets2_pt, "W": w_pt}
        model.run_with_tensors(inputs, [y, z])

        torch.testing.assert_close(y, x_pt)
        torch.testing.assert_close(z, z_pt)

    def test_make_jagged(self):
        self._test_make_jagged(
            check_sequence_lengths=True,
            test_name="test_make_jagged",
        )

    def test_make_jagged_no_seq_len_check(self):
        self._test_make_jagged(
            check_sequence_lengths=False,
            test_name="test_make_jagged_no_seq_len_check",
        )

    def test_make_jagged_with_dynamic_bounds(
        self,
        dtype="float16",
        offsets_dtype="int32",
    ):
        B = 4
        N_min = 1
        N_max = 32
        N = 3
        D = 64

        batch_dim = IntVar(name="batch_size", values=[1, B])
        max_seq_dim = IntVar(name="max_seq_len", values=[N_min, N_max])
        embedding_dim = IntImm(name="embedding", value=D)

        total_length_dim = IntVar(name="total_length", values=[0, B * N_max])
        offsets_dim = IntVar(name="offsets_size", values=[2, B + 1])

        SOURCE = Tensor(
            shape=[
                total_length_dim,
                embedding_dim,
            ],
            name="source",
            dtype=dtype,
            is_input=True,
        )
        OFFSETS_LIST = [
            Tensor(
                shape=[
                    offsets_dim,
                ],
                name="offsets",
                dtype=offsets_dtype,
                is_input=True,
            )
        ]
        DENSE = Tensor(
            shape=[
                batch_dim,
                max_seq_dim,
                embedding_dim,
            ],
            name="dense",
            dtype=dtype,
            is_input=True,
        )

        JAGGED = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=[
                JaggedDim(
                    min_value=0,
                    max_value=max_seq_dim,
                )
            ],
        )(
            source=SOURCE,
            offsets_list=OFFSETS_LIST,
        )

        RESULT = ops.elementwise(FuncEnum.ADD)(JAGGED, DENSE)

        assert not SOURCE.is_jagged()
        assert not DENSE.is_jagged()
        assert JAGGED.is_jagged()
        assert RESULT.is_jagged()

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        model = compile_model(
            [RESULT],
            detect_target(),
            "./tmp",
            "test_make_jagged_with_dynamic_bounds",
        )

        offsets = [0, 1, 4, 6, 7]
        torch_offsets_type = string_to_torch_dtype(offsets_dtype)
        offsets_pt = torch.tensor(offsets, dtype=torch_offsets_type).cuda()
        source_pt = get_random_torch_tensor([offsets[-1], D], dtype=dtype)
        dense_pt = get_random_torch_tensor([B, N, D], dtype=dtype)

        result_pt = add_jagged_dense_ref(
            jagged=source_pt,
            offsets_list=[offsets_pt],
            jagged_max_shape=[B, N, D],
            dense=dense_pt,
        )
        result = torch.empty_like(result_pt)

        inputs = {"source": source_pt, "offsets": offsets_pt, "dense": dense_pt}
        model.run_with_tensors(inputs, [result])

        torch.testing.assert_close(result, result_pt)

    def test_make_jagged_multiple_sources(
        self,
        num_sources=3,
        dtype="float16",
        offsets_dtype="int32",
    ):
        B = 4
        N = 3
        D = 64

        batch_dim = IntVar(name="batch_size", values=[1, B])
        max_seq_dim = IntImm(name="max_seq_len", value=N)
        embedding_dim = IntImm(name="embedding", value=D)

        total_length_dim = IntVar(name="total_length", values=[0, B * N])
        offsets_dim = IntVar(name="offsets_size", values=[2, B + 1])

        SOURCES = [
            Tensor(
                shape=[
                    total_length_dim,
                    embedding_dim,
                ],
                name=f"source_{i}",
                dtype=dtype,
                is_input=True,
            )
            for i in range(num_sources)
        ]
        OFFSETS_LIST = [
            Tensor(
                shape=[
                    offsets_dim,
                ],
                name="offsets",
                dtype=offsets_dtype,
                is_input=True,
            )
        ]
        DENSE = Tensor(
            shape=[
                batch_dim,
                max_seq_dim,
                embedding_dim,
            ],
            name="dense",
            dtype=dtype,
            is_input=True,
        )

        JAGGEDS = ops.make_jagged(
            batch_dim=batch_dim,
            jagged_dims=[
                JaggedDim(
                    min_value=0,
                    max_value=max_seq_dim,
                )
            ],
        )(
            source=SOURCES,
            offsets_list=OFFSETS_LIST,
        )

        RESULT = DENSE
        for JAGGED in JAGGEDS:
            RESULT = ops.elementwise(FuncEnum.ADD)(JAGGED, RESULT)

        assert all(not SOURCE.is_jagged() for SOURCE in SOURCES)
        assert not DENSE.is_jagged()
        assert all(JAGGED.is_jagged() for JAGGED in JAGGEDS)
        assert RESULT.is_jagged()

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        model = compile_model(
            [RESULT],
            detect_target(),
            "./tmp",
            "test_make_jagged_multiple_sources",
        )

        offsets = [0, 1, 4, 6, 7]
        torch_offsets_type = string_to_torch_dtype(offsets_dtype)
        offsets_pt = torch.tensor(offsets, dtype=torch_offsets_type).cuda()
        sources_pt = {
            f"source_{i}": get_random_torch_tensor([offsets[-1], D], dtype=dtype)
            for i in range(num_sources)
        }
        dense_pt = get_random_torch_tensor([B, N, D], dtype=dtype)

        sources_list_pt = list(sources_pt.values())
        summed_sources_pt = torch.clone(sources_list_pt[0])
        for source_pt in sources_list_pt[1:]:
            summed_sources_pt += source_pt
        result_pt = add_jagged_dense_ref(
            jagged=summed_sources_pt,
            offsets_list=[offsets_pt],
            jagged_max_shape=[B, N, D],
            dense=dense_pt,
        )
        result = torch.empty_like(result_pt)

        inputs = {**sources_pt, "offsets": offsets_pt, "dense": dense_pt}
        model.run_with_tensors(inputs, [result])

        torch.testing.assert_close(result, result_pt, rtol=1e-2, atol=1e-2)

    def test_make_jagged_dedup(
        self,
        dtype="float16",
        offsets_dtype="int32",
    ):
        B = 4
        N = 3
        D = 64
        W = 32

        batch_dim = IntVar(name="batch_size", values=[1, B])
        max_seq_dim = IntImm(name="max_seq_len", value=N)
        embedding_dim = IntImm(name="embedding", value=D)
        weights_dim = IntImm(name="weight", value=W)

        total_length_dim = IntVar(name="total_length", values=[0, B * N])
        offsets_dim = IntVar(name="offsets_size", values=[2, B + 1])
        jagged_dims = [JaggedDim(min_value=0, max_value=max_seq_dim)]
        num_sources = 4

        X1, X2, X3, X4 = [
            Tensor(
                shape=[
                    total_length_dim,
                    embedding_dim,
                ],
                name=f"x_{i}",
                dtype=dtype,
                is_input=True,
            )
            for i in range(num_sources)
        ]
        OFFSETS_LIST = [
            Tensor(
                shape=[
                    offsets_dim,
                ],
                name="offsets",
                dtype=offsets_dtype,
                is_input=True,
            )
        ]
        DENSE = Tensor(
            shape=[
                batch_dim,
                max_seq_dim,
                weights_dim,
            ],
            name="dense",
            dtype=dtype,
            is_input=True,
        )
        WEIGHTS = Tensor(
            shape=[
                embedding_dim,
                weights_dim,
            ],
            name="weights",
            dtype=dtype,
            is_input=True,
        )

        Y1, Y2 = (
            ops.make_jagged(batch_dim=batch_dim, jagged_dims=jagged_dims)(
                source=SOURCE,
                offsets_list=OFFSETS_LIST,
            )
            for SOURCE in (X1, X2)
        )
        Y3, Y4 = (ops.gemm_rrr()(SOURCE, WEIGHTS) for SOURCE in (X3, X4))
        Z1, Z2 = (ops.gemm_rrr()(SOURCE, WEIGHTS) for SOURCE in (Y1, Y2))
        Z3, Z4 = (
            ops.make_jagged(batch_dim=batch_dim, jagged_dims=jagged_dims)(
                source=SOURCE,
                offsets_list=OFFSETS_LIST,
            )
            for SOURCE in (Y3, Y4)
        )
        RESULT = DENSE
        for Z in (Z1, Z2, Z3, Z4):
            RESULT = ops.elementwise(FuncEnum.ADD)(RESULT, Z)

        RESULT._attrs["name"] = "result"
        RESULT._attrs["is_output"] = True

        for X in (X1, X2, X3, X4):
            assert not X.is_jagged()
        assert Y1.is_jagged()
        assert Y2.is_jagged()
        assert not Y3.is_jagged()
        assert not Y4.is_jagged()
        for Z in (Z1, Z2, Z3, Z4):
            assert Z.is_jagged()
        assert not DENSE.is_jagged()
        assert RESULT.is_jagged()

        model = compile_model(
            [RESULT],
            detect_target(),
            "./tmp",
            "test_make_jagged_dedup",
        )

        make_jagged_ops = [
            op
            for op in get_sorted_ops(model.debug_sorted_graph)
            if op._attrs["op"] == "make_jagged"
        ]
        assert len(make_jagged_ops) == 1
        make_jagged_inputs = set(make_jagged_ops[0]._attrs["inputs"])
        assert make_jagged_ops[0]._attrs["num_sources"] == num_sources
        for X in (X1, X2, X3, X4):
            assert not X.is_jagged()
            assert X in make_jagged_inputs
        assert OFFSETS_LIST[0] in make_jagged_inputs
        for Y in (Y1, Y2, Y3, Y4):
            assert Y.is_jagged()
        for Z in (Z1, Z2, Z3, Z4):
            assert Z.is_jagged()
        assert not DENSE.is_jagged()
        assert RESULT.is_jagged()

        offsets = [0, 1, 4, 6, 7]
        torch_offsets_type = string_to_torch_dtype(offsets_dtype)
        offsets_pt = torch.tensor(offsets, dtype=torch_offsets_type).cuda()
        xs_pt = {
            f"x_{i}": get_random_torch_tensor([offsets[-1], D], dtype=dtype)
            for i in range(num_sources)
        }
        weights_pt = get_random_torch_tensor([D, W], dtype=dtype)
        dense_pt = get_random_torch_tensor([B, N, W], dtype=dtype)

        ys_pt = [torch.matmul(x_pt, weights_pt) for x_pt in xs_pt.values()]
        summed_ys_pt = torch.clone(ys_pt[0])
        for y_pt in ys_pt[1:]:
            summed_ys_pt += y_pt
        result_pt = add_jagged_dense_ref(
            jagged=summed_ys_pt,
            offsets_list=[offsets_pt],
            jagged_max_shape=[B, N, W],
            dense=dense_pt,
        )

        inputs = {
            **xs_pt,
            "offsets": offsets_pt,
            "dense": dense_pt,
            "weights": weights_pt,
        }
        result = torch.empty_like(result_pt)
        model.run_with_tensors(inputs, [result])

        torch.testing.assert_close(result, result_pt, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
