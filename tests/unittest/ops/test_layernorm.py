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
Unittests for LayerNorm Operator.
"""
import logging
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.torch_utils import string_to_torch_dtype


class LayernormTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LayernormTestCase, self).__init__(*args, **kwargs)
        self.test_count = 0

    def _test_layernorm(
        self,
        MS=(),
        NS=(1496,),
        gamma_is_none=False,
        beta_is_none=False,
        use_size_op=False,
        eps=1e-5,
        atol=1e-3,
        rtol=1e-3,
        dtype="float16",
    ):
        torch_dtype = string_to_torch_dtype(dtype)
        BS = [1, 1024]
        input_shapes = ((BS), *MS, *NS)
        logging.info(
            f"input shapes: {input_shapes}"
            f"gamma_is_none: {gamma_is_none}, beta_is_none: {beta_is_none}, "
            f"use_size_op: {use_size_op}"
            f"dtype: {dtype}"
        )
        assert isinstance(MS, (list, tuple))
        assert isinstance(NS, (list, tuple))

        X1 = Tensor(
            shape=[IntVar(name="input_batch", values=BS), *MS, *NS],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        if gamma_is_none:
            X2 = None
        else:
            X2 = Tensor(
                shape=NS,
                dtype=dtype,
                name="gamma",
                is_input=True,
            )
        if beta_is_none:
            X3 = None
        else:
            X3 = Tensor(
                shape=NS,
                dtype=dtype,
                name="beta",
                is_input=True,
            )
        if use_size_op:
            norm_shapes = [
                ops.getitem()(ops.size()(X1), i) for i in range(1 + len(MS), X1._rank())
            ]
        else:
            norm_shapes = [IntImm(n) for n in NS]

        X4 = (
            ops.layernorm()(X1, X2, X3, NS, eps)
            if not use_size_op
            else ops.layernorm()(X1, X2, X3, norm_shapes, eps)
        )
        X4._attrs["is_output"] = True
        X4._attrs["name"] = "output"

        target = detect_target()
        dll_name = f"test_{self.test_count}.so"
        module = compile_model(X4, target, "./tmp", "layernorm", dll_name=dll_name)

        for batch_size in [50, 900, 1024]:
            x1_pt = torch.randn(batch_size, *MS, *NS, dtype=torch_dtype).cuda()
            if gamma_is_none:
                x2_pt = None
            else:
                x2_pt = torch.randn(NS, dtype=torch_dtype).cuda()
            if beta_is_none:
                x3_pt = None
            else:
                x3_pt = torch.randn(NS, dtype=torch_dtype).cuda()
            x4_pt = torch.nn.functional.layer_norm(x1_pt, NS, x2_pt, x3_pt, eps=eps)

            inputs = {"X": x1_pt}
            if not gamma_is_none:
                inputs["gamma"] = x2_pt
            if not beta_is_none:
                inputs["beta"] = x3_pt
            x4 = torch.empty([batch_size, *MS, *NS], dtype=torch_dtype).cuda()
            module.run_with_tensors(inputs, [x4])
            torch.testing.assert_close(x4, x4_pt, atol=atol, rtol=rtol)
            self.test_count += 1

    def test_layernorm(self):
        if detect_target().name() == "rocm":
            self._test_layernorm(use_size_op=False, MS=(256,), NS=(768,))
            self._test_layernorm(use_size_op=False, MS=(), NS=(768,))
            self._test_layernorm(
                use_size_op=False,
                MS=(
                    256,
                    3,
                ),
                NS=(256,),
            )
        else:
            for use_size_op in (True, False):
                self._test_layernorm(use_size_op=use_size_op)
                self._test_layernorm(gamma_is_none=True, use_size_op=use_size_op)
                self._test_layernorm(beta_is_none=True, use_size_op=use_size_op)
                self._test_layernorm(
                    gamma_is_none=True, beta_is_none=True, use_size_op=use_size_op
                )
                self._test_layernorm(use_size_op=use_size_op, eps=0.1)
                self._test_layernorm(MS=(16, 64), NS=(4, 32), use_size_op=use_size_op)
                self._test_layernorm(
                    MS=(16, 8, 4), NS=(2, 4, 32), use_size_op=use_size_op
                )

    @unittest.skipIf(
        detect_target().name() == "rocm", "fp32 layer norm is not supported on ROCm"
    )
    def test_layernorm_fp32(self):
        self._test_layernorm(dtype="float32")
        self._test_layernorm(gamma_is_none=True, dtype="float32")
        self._test_layernorm(beta_is_none=True, dtype="float32")
        self._test_layernorm(gamma_is_none=True, beta_is_none=True, dtype="float32")
        self._test_layernorm(eps=0.1, dtype="float32")
        self._test_layernorm(MS=(16, 64), NS=(4, 32), dtype="float32")
        self._test_layernorm(MS=(16, 8, 4), NS=(2, 4, 32), dtype="float32")

    @unittest.skipIf(
        detect_target().name() == "rocm", "fp32 layer norm is not supported on ROCm"
    )
    def test_layernorm_bf16(self):
        self._test_layernorm(dtype="bfloat16", atol=1e-2, rtol=1e-2)
        self._test_layernorm(gamma_is_none=True, dtype="bfloat16", atol=1e-2, rtol=1e-2)
        self._test_layernorm(beta_is_none=True, dtype="bfloat16", atol=1e-2, rtol=1e-2)
        self._test_layernorm(
            gamma_is_none=True,
            beta_is_none=True,
            dtype="bfloat16",
            atol=1e-2,
            rtol=1e-2,
        )
        self._test_layernorm(eps=0.1, dtype="bfloat16", atol=1e-2, rtol=1e-2)
        self._test_layernorm(
            MS=(16, 64), NS=(4, 32), dtype="bfloat16", atol=1e-2, rtol=1e-2
        )
        self._test_layernorm(
            MS=(16, 8, 4), NS=(2, 4, 32), dtype="bfloat16", atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
