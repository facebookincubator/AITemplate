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
from typing import Callable, List

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ClampTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def _create_shape_from_list(self, shape: List[int]) -> IntVar:
        if len(shape) > 1:
            return IntVar(shape)
        return IntImm(shape[0])

    def _float_to_tensor(
        self,
        name: str,
        value: float,
        dtype="float16",
    ) -> Tensor:
        return Tensor(
            shape=[],
            dtype=dtype,
            name=name,
            value=value,
        )

    def _test_helper(
        self,
        input_shape: List[List[int]],
        arg_a: float,
        arg_b: float,
        arg_c: float,
        add_nans: bool,
        add_infs: bool,
        test_name: str,
        func: FuncEnum,
        get_expected: Callable[[torch.Tensor], torch.Tensor],
        dtype="float16",
    ):
        self.assertGreater(len(input_shape), 0)
        X = Tensor(
            shape=[self._create_shape_from_list(shape) for shape in input_shape],
            dtype=dtype,
            name="input",
            is_input=True,
        )
        a_tensor = self._float_to_tensor("a", arg_a, dtype=dtype)
        b_tensor = self._float_to_tensor("b", arg_b, dtype=dtype)
        c_tensor = self._float_to_tensor("c", arg_c, dtype=dtype)

        result = ops.elementwise(func)(X, a_tensor, b_tensor, c_tensor)
        result._attrs["is_output"] = True
        result._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(result, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1

        torch_dtype = string_to_torch_dtype(dtype)
        for shape in itertools.product(*input_shape):
            X_pt = get_random_torch_tensor(shape, dtype=dtype)
            if add_nans:
                X_pt[0].fill_(float("nan"))
            if add_infs:
                X_pt[1].fill_(float("inf"))
                X_pt[2].fill_(-float("inf"))

            actual = torch.empty_like(X_pt)
            module.run_with_tensors([X_pt], [actual])
            expected = get_expected(X_pt, torch_dtype)
            torch.testing.assert_close(expected, actual)

    def _test_nan_to_num(
        self,
        input_shape: List[List[int]],
        nan_replacement: float,
        inf_replacement: float,
        neginf_replacement: float,
        add_nans: bool = False,
        add_infs: bool = False,
        test_name: str = "nan_to_num",
        dtype="float16",
    ):
        nan_to_num_pt = (
            lambda x, torch_dtype: x.to(torch.float)
            .nan_to_num(
                posinf=inf_replacement,
                neginf=neginf_replacement,
                nan=nan_replacement,
            )
            .to(torch_dtype)
        )
        self._test_helper(
            input_shape=input_shape,
            arg_a=nan_replacement,
            arg_b=inf_replacement,
            arg_c=neginf_replacement,
            add_nans=add_nans,
            add_infs=add_infs,
            test_name=test_name,
            func=FuncEnum.NAN_TO_NUM,
            get_expected=nan_to_num_pt,
            dtype=dtype,
        )

    def _test_clamp_nan_to_num(
        self,
        input_shape: List[List[int]],
        clamp_min: float,
        clamp_max: float,
        nan_replacement: float,
        add_nans: bool = False,
        test_name: str = "clamp_nan_to_num",
        dtype="float16",
    ):
        clamp_nan_to_num_pt = (
            lambda x, torch_dtype: x.to(torch.float)
            .clamp(clamp_min, clamp_max)
            .nan_to_num(nan=nan_replacement)
            .to(torch_dtype)
        )
        self._test_helper(
            input_shape=input_shape,
            arg_a=clamp_min,
            arg_b=clamp_max,
            arg_c=nan_replacement,
            add_nans=add_nans,
            add_infs=False,
            test_name=test_name,
            func=FuncEnum.CLAMP_NAN_TO_NUM,
            get_expected=clamp_nan_to_num_pt,
            dtype=dtype,
        )

    def test_clamp_nan_to_num_fp16(self):
        clamp_arg_sets = [(-1.0, 2.0, 0.0), (-42.0, 2.0, 43.0)]
        for clamp_min, clamp_max, nan_replacement in clamp_arg_sets:
            self._test_clamp_nan_to_num(
                input_shape=[[40, 2], [40], [40]],
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                nan_replacement=nan_replacement,
                add_nans=False,
                test_name="clamp_nan_to_num_fp16",
                dtype="float16",
            )
            self._test_clamp_nan_to_num(
                input_shape=[[40, 3], [3], [3]],
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                nan_replacement=nan_replacement,
                add_nans=True,
                test_name="clamp_nan_to_num_fp16",
                dtype="float16",
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_clamp_nan_to_num_fp32(self):
        clamp_arg_sets = [(-1.0, 2.0, 0.0), (-42.0, 2.0, 43.0)]
        for clamp_min, clamp_max, nan_replacement in clamp_arg_sets:
            self._test_clamp_nan_to_num(
                input_shape=[[40, 2], [40], [40]],
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                nan_replacement=nan_replacement,
                add_nans=False,
                test_name="clamp_nan_to_num_fp32",
                dtype="float32",
            )
            self._test_clamp_nan_to_num(
                input_shape=[[40, 3], [3], [3]],
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                nan_replacement=nan_replacement,
                add_nans=True,
                test_name="clamp_nan_to_num_fp32",
                dtype="float32",
            )

    def test_nan_to_num_fp16(self):
        clamp_arg_sets = [(-1.0, 2.0, 0.0), (-42.0, 2.0, 43.0)]
        for nan_replacement, inf_replacement, neginf_replacement in clamp_arg_sets:
            self._test_nan_to_num(
                input_shape=[[40, 2], [40], [40]],
                nan_replacement=nan_replacement,
                inf_replacement=inf_replacement,
                neginf_replacement=neginf_replacement,
                add_nans=False,
                add_infs=False,
                test_name="nan_to_num_fp16",
                dtype="float16",
            )
            self._test_nan_to_num(
                input_shape=[[40, 3], [3], [3]],
                nan_replacement=nan_replacement,
                inf_replacement=inf_replacement,
                neginf_replacement=neginf_replacement,
                add_nans=True,
                add_infs=True,
                test_name="nan_to_num_fp16",
                dtype="float16",
            )
            self._test_nan_to_num(
                input_shape=[[40, 3], [3], [3]],
                nan_replacement=float("inf"),
                inf_replacement=inf_replacement,
                neginf_replacement=neginf_replacement,
                add_nans=True,
                add_infs=True,
                test_name="nan_to_num_fp16",
                dtype="float16",
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_nan_to_num_fp32(self):
        clamp_arg_sets = [(-1.0, 2.0, 0.0), (-42.0, 2.0, 43.0)]
        for nan_replacement, inf_replacement, neginf_replacement in clamp_arg_sets:
            self._test_nan_to_num(
                input_shape=[[40, 2], [40], [40]],
                nan_replacement=nan_replacement,
                inf_replacement=inf_replacement,
                neginf_replacement=neginf_replacement,
                add_nans=False,
                add_infs=False,
                test_name="nan_to_num_fp32",
                dtype="float32",
            )
            self._test_nan_to_num(
                input_shape=[[40, 3], [3], [3]],
                nan_replacement=nan_replacement,
                inf_replacement=inf_replacement,
                neginf_replacement=neginf_replacement,
                add_nans=True,
                add_infs=True,
                test_name="nan_to_num_fp32",
                dtype="float32",
            )
            self._test_nan_to_num(
                input_shape=[[40, 3], [3], [3]],
                nan_replacement=float("inf"),
                inf_replacement=inf_replacement,
                neginf_replacement=neginf_replacement,
                add_nans=True,
                add_infs=True,
                test_name="nan_to_num_fp32",
                dtype="float32",
            )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
