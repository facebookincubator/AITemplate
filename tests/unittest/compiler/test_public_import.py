# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

from aitemplate.compiler.public import *  # noqa: F403


class PublicImportTestCase(unittest.TestCase):
    """Tests whether compiler.ops.public imports classes correctly."""

    def test_import(self):
        input1 = Tensor(shape=[IntVar([1024, 2048]), IntImm(32)])  # noqa: F405
        input2 = Tensor(shape=[IntImm(32), IntImm(128)])  # noqa: F405

        o1 = elementwise(FuncEnum.ADD)(input1, input1)  # noqa: F405

        _ = gemm_rrr()(input1, input2)  # noqa: F405

        x1 = reshape()(input1, [1, -1, 32])  # noqa: F405
        x2 = reshape()(input2, [1, 32, 128])  # noqa: F405
        o3 = bmm_rrr()(x1, x2)  # noqa: F405

        _ = reduce_sum(dim=2)(o3)  # noqa: F405

        _ = concatenate()([input1, o1], dim=1)  # noqa: F405

        _ = permute()(o3, (0, 2, 1))  # noqa: F405

        _ = topk(k=10)(input1)  # noqa: F405
        _ = layernorm(input1)  # noqa: F405

        _ = clamp()(input1, -1, 1)  # noqa: F405

        _ = reduce_mean(dim=1)(input1)  # noqa: F405
        _ = vector_norm(ord_kind=2, dim=0)(input1)  # noqa: F405
        _ = var(dim=0, unbiased=False, keepdim=True)(input1)  # noqa: F405
        _ = softmax()(input1, dim=1)  # noqa: F405


if __name__ == "__main__":
    unittest.main()
