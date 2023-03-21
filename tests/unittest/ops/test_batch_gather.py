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
"""
Unittests for batch_gather Operator.
"""
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype


class gatherTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_count = 0

    def _create_tensors(self, N, dtype):
        scores = torch.randperm(N) / N
        return scores.cuda().to(dtype=string_to_torch_dtype(dtype))


class batchGatherTestCase(gatherTestCase):
    def _test_batch_gather(
        self,
        shape=(3, 2, 2),
        ind_shape=(3,),
        dim=0,
        max_ind=2,
        test_name="gather",
        dtype="float16",
    ):
        in_shape = shape

        o_shape = list(in_shape)
        rank = len(ind_shape)
        o_shape[rank - 1] = ind_shape[-1]

        X1 = Tensor(
            shape=in_shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        X2 = Tensor(
            shape=ind_shape,
            dtype="int64",
            name="indices",
            is_input=True,
        )
        X4 = ops.batch_gather()(X1, X2)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(X4, target, "./tmp", f"{test_name}_{self.test_count}")

        input_x = get_random_torch_tensor(in_shape, dtype)
        init_index = torch.randint(max_ind, size=ind_shape, dtype=torch.int64).cuda()

        reshaped_shape = list(ind_shape)
        for _ in range(len(input_x.shape) - len(ind_shape)):
            reshaped_shape.append(1)

        gather_index = torch.reshape(init_index, reshaped_shape)
        gather_index = torch.broadcast_to(gather_index, o_shape)

        y_pt = torch.gather(input_x, dim, gather_index)

        x = input_x.reshape(in_shape).contiguous()

        indices = init_index.reshape(ind_shape).contiguous()

        y = torch.empty_like(y_pt)
        module.run_with_tensors({"X": x, "indices": indices}, [y])

        torch.testing.assert_close(y_pt, y, atol=1e-2, rtol=1e-2)

    def test_batch_gather(self):
        self._test_batch_gather(
            shape=(8, 2, 2), ind_shape=(2,), dim=0, max_ind=8, test_name="batch_gather1"
        )
        self._test_batch_gather(
            shape=(2, 2), ind_shape=(2, 2), dim=1, max_ind=2, test_name="batch_gather2"
        )
        self._test_batch_gather(
            shape=(2, 2), ind_shape=(2, 1), dim=1, max_ind=2, test_name="batch_gather3"
        )
        self._test_batch_gather(
            shape=(8, 4, 4, 2, 2),
            ind_shape=(8, 4, 1),
            dim=2,
            max_ind=4,
            test_name="batch_gather4",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported by ROCM.")
    def test_float32(self):
        self._test_batch_gather(
            shape=(8, 2, 2),
            ind_shape=(2,),
            dim=0,
            max_ind=8,
            test_name="batch_gather_f32",
            dtype="float32",
        )


class batchGatherTopkTestCase(gatherTestCase):
    def _test_batch_gather_topk(
        self,
        shape=(2, 2, 2),
        batch_size=1,
        N=1000,
        topK=100,
        test_name="topk",
        dtype="float16",
    ):
        m_shape = (N,) + shape
        n_shape = (topK,) + shape

        X1 = Tensor(
            shape=m_shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        X2 = Tensor(
            shape=[N],
            dtype=dtype,
            name="scores",
            is_input=True,
        )
        X3 = ops.topk(k=topK)(X2)
        X4 = ops.batch_gather()(X1, X3)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(X4, target, "./tmp", f"{test_name}_{self.test_count}")

        input_x = get_random_torch_tensor(m_shape, dtype)
        scores = self._create_tensors(N, dtype)

        (_, init_index) = torch.topk(scores, k=topK, dim=0)

        reshaped_shape = [topK]
        for _ in range(len(input_x.shape) - 1):
            reshaped_shape.append(1)

        gather_index = torch.reshape(init_index, reshaped_shape)
        gather_index = torch.broadcast_to(gather_index, n_shape)

        y_pt = torch.gather(input_x, 0, gather_index)

        x = input_x.reshape(m_shape).contiguous()

        x_scores = scores.reshape((N,)).contiguous()

        y = torch.empty_like(y_pt)
        module.run_with_tensors({"X": x, "scores": x_scores}, [y])

        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_batch_gather_topk(self):
        self._test_batch_gather_topk(
            shape=(4, 1, 1), N=2000, topK=300, test_name="batch_gather_topk"
        )

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported by ROCM.")
    def test_float32(self):
        self._test_batch_gather_topk(
            shape=(4, 1, 1),
            N=2000,
            topK=300,
            test_name="batch_gather_topk_f32",
            dtype="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
