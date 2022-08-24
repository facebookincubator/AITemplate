# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Unittests for topk Operator.
"""
import unittest

import torch

from aitemplate.compiler import ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


class gatherTestCase(unittest.TestCase):
    def _create_tensors(self, N):
        scores = torch.randperm(N) / N
        return scores.cuda().half()


class batchGatherTestCase(gatherTestCase):
    def _create_tensors(self, N):
        scores = torch.randperm(N) / N
        return scores.cuda().half()

    def _test_batch_gather(
        self, shape=(3, 2, 2), ind_shape=(3,), dim=0, max_ind=2, test_name="gather"
    ):

        in_shape = shape

        o_shape = list(in_shape)
        rank = len(ind_shape)
        o_shape[rank - 1] = ind_shape[-1]

        X1 = Tensor(
            shape=in_shape,
            dtype="float16",
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
        module = gen_execution_module(X4, target, "./tmp", test_name)

        input_x = torch.rand(in_shape).cuda().half()
        init_index = torch.randint(max_ind, size=ind_shape, dtype=torch.int64).cuda()

        reshaped_shape = list(ind_shape)
        for _ in range(len(input_x.shape) - len(ind_shape)):
            reshaped_shape.append(1)

        gather_index = torch.reshape(init_index, reshaped_shape)
        gather_index = torch.broadcast_to(gather_index, o_shape)

        y_pt = torch.gather(input_x, dim, gather_index)

        x = input_x.reshape(in_shape).contiguous()

        indices = init_index.reshape(ind_shape).contiguous()

        y = torch.empty(o_shape).cuda().half()
        module.RunWithTensors({"X": x, "indices": indices}, [y])

        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

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


class batchGatherTopkTestCase(gatherTestCase):
    def _test_batch_gather_topk(
        self, shape=(2, 2, 2), batch_size=1, N=1000, topK=100, test_name="topk"
    ):

        m_shape = (N,) + shape
        n_shape = (topK,) + shape

        X1 = Tensor(
            shape=m_shape,
            dtype="float16",
            name="X",
            is_input=True,
        )
        X2 = Tensor(
            shape=[N],
            dtype="float16",
            name="scores",
            is_input=True,
        )
        X3 = ops.topk(k=topK)(X2)
        X4 = ops.batch_gather()(X1, X3)
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        module = gen_execution_module(X4, target, "./tmp", test_name)

        input_x = torch.rand(m_shape).cuda().half()
        scores = self._create_tensors(N)

        (_, init_index) = torch.topk(scores, k=topK, dim=0)

        reshaped_shape = [topK]
        for _ in range(len(input_x.shape) - 1):
            reshaped_shape.append(1)

        gather_index = torch.reshape(init_index, reshaped_shape)
        gather_index = torch.broadcast_to(gather_index, n_shape)

        y_pt = torch.gather(input_x, 0, gather_index)

        x = input_x.reshape(m_shape).contiguous()

        x_scores = scores.reshape((N,)).contiguous()

        y = torch.empty(n_shape).cuda().half()
        module.RunWithTensors({"X": x, "scores": x_scores}, [y])

        self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_batch_gather_topk(self):
        self._test_batch_gather_topk(
            shape=(4, 1, 1), N=2000, topK=300, test_name="batch_gather_topk"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
