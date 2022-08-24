# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import itertools
import unittest
from typing import Callable, List

import torch

from aitemplate.compiler import ops
from aitemplate.compiler.base import IntImm, IntVar
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target, gen_execution_module


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class ClampTestCase(unittest.TestCase):
    def _create_shape_from_list(self, shape: List[int]) -> IntVar:
        if len(shape) > 1:
            return IntVar(shape)
        return IntImm(shape[0])

    def _float_to_tensor(self, name: str, value: float) -> Tensor:
        return Tensor(shape=[], dtype="float16", name=name, value=value)

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
    ):
        self.assertGreater(len(input_shape), 0)
        X = Tensor(
            shape=[self._create_shape_from_list(shape) for shape in input_shape],
            dtype="float16",
            name="input",
            is_input=True,
        )
        a_tensor = self._float_to_tensor("a", arg_a)
        b_tensor = self._float_to_tensor("b", arg_b)
        c_tensor = self._float_to_tensor("c", arg_c)

        result = ops.elementwise(func)(X, a_tensor, b_tensor, c_tensor)
        result._attrs["is_output"] = True
        result._attrs["name"] = "output"

        target = detect_target()
        module = gen_execution_module(result, target, "./tmp", test_name)

        for shape in itertools.product(*input_shape):
            X_pt = torch.randn(shape, dtype=torch.half).cuda()
            if add_nans:
                X_pt[0].fill_(float("nan"))
            if add_infs:
                X_pt[1].fill_(float("inf"))
                X_pt[2].fill_(-float("inf"))

            actual = torch.empty(shape).cuda().half()
            module.RunWithTensors([X_pt], [actual])

            expected = get_expected(X_pt).cuda()
            self.assertTrue(torch.equal(expected, actual))

    def _test_nan_to_num(
        self,
        test_num: int,
        input_shape: List[List[int]],
        nan_replacement: float,
        inf_replacement: float,
        neginf_replacement: float,
        add_nans: bool = False,
        add_infs: bool = False,
    ):
        nan_to_num_pt = (
            lambda x: x.to(torch.float)
            .nan_to_num(
                posinf=inf_replacement, neginf=neginf_replacement, nan=nan_replacement
            )
            .to(torch.half)
        )
        self._test_helper(
            input_shape,
            nan_replacement,
            inf_replacement,
            neginf_replacement,
            add_nans,
            add_infs,
            f"nan_to_num_{test_num}",
            FuncEnum.NAN_TO_NUM,
            nan_to_num_pt,
        )

    def _test_clamp_nan_to_num(
        self,
        test_num: int,
        input_shape: List[List[int]],
        clamp_min: float,
        clamp_max: float,
        nan_replacement: float,
        add_nans: bool = False,
    ):
        clamp_nan_to_num_pt = (
            lambda x: x.to(torch.float)
            .clamp(clamp_min, clamp_max)
            .nan_to_num(nan=nan_replacement)
            .to(torch.half)
        )
        self._test_helper(
            input_shape,
            clamp_min,
            clamp_max,
            nan_replacement,
            add_nans,
            False,
            f"clamp_nan_to_num_{test_num}",
            FuncEnum.CLAMP_NAN_TO_NUM,
            clamp_nan_to_num_pt,
        )

    def test_clamp_nan_to_num(self):
        clamp_arg_sets = [(-1.0, 2.0, 0.0), (-42.0, 2.0, 43.0)]
        test_num = 0
        for clamp_args in clamp_arg_sets:
            self._test_clamp_nan_to_num(
                test_num,
                [[40, 2], [40], [40]],
                *clamp_args,
                add_nans=False,
            )
            self._test_clamp_nan_to_num(
                test_num + 1,
                [[40, 3], [3], [3]],
                *clamp_args,
                add_nans=True,
            )
            test_num += 2

    def test_nan_to_num(self):
        clamp_arg_sets = [(-1.0, 2.0, 0.0), (-42.0, 2.0, 43.0)]
        test_num = 0
        for clamp_args in clamp_arg_sets:
            self._test_nan_to_num(
                test_num,
                [[40, 2], [40], [40]],
                *clamp_args,
                add_nans=False,
                add_infs=False,
            )
            self._test_nan_to_num(
                test_num + 1,
                [[40, 3], [3], [3]],
                *clamp_args,
                add_nans=True,
                add_infs=True,
            )
            test_num += 2


if __name__ == "__main__":
    unittest.main()
