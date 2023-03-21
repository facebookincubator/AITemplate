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
Operator definition for layernorm.
"""
import logging
import os
import re
from collections import OrderedDict
from hashlib import sha1
from typing import Any, List, Union

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.compiler.base import (
    DynamicProfileStrategy,
    ExecItem,
    IntImm,
    IntVar,
    Operator,
    Tensor,
)
from aitemplate.compiler.ops.softmax.cache_entry import NormQueryEntry, NormRecordEntry
from aitemplate.compiler.tensor_accessor import TensorAccessor

from aitemplate.testing import detect_target
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,W0223


_LOGGER = logging.getLogger(__name__)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class layernorm(Operator):
    """Standalone layernorm op.
    Applies Layer Normalization over a mini-batch of inputs as described in the
    paper Layer Normalization. The mean and standard-deviation are calculated
    over the last D dimensions, where D is the dimension of normalized_shape.
    Input shape: [M0, M1, ..., Mp, N1, N2, ..., ND]
    Normalized_shape: [N1, N2, ..., ND]
    Gamma/Beta, if not None, have the same shape as normalized_shape.
    """

    def __init__(self, normalized_shape: List[IntImm] = None) -> None:
        super().__init__()
        self._attrs["op"] = "layernorm"
        self._attrs["has_profiler"] = False
        if detect_target().name() == "rocm":
            self._attrs["has_profiler"] = True
        self._attrs["default_normalized_shape"] = normalized_shape
        self._attrs["normalized_shape"] = []

    @staticmethod
    def check_shapes(x_shapes, gamma_shapes, beta_shapes, normalized_shape):
        if len(normalized_shape) >= len(x_shapes):
            raise NotImplementedError(
                f"Layernorm normalized_shape length must be smaller than the input."
                f"Current normalized_shape: {normalized_shape}, input shape: {x_shapes}"
            )

        def _check_param_shapes(x_shapes, param_shapes, param_name):
            if param_name != "normalized" and not param_shapes:
                return
            for shape in param_shapes:
                if not isinstance(shape, IntImm):
                    raise NotImplementedError(
                        f"Layernorm {param_name} shape must be immutable values."
                        f"Current value: {param_shapes}"
                    )

            batch_ndims = len(x_shapes) - len(param_shapes)
            for i in range(len(param_shapes)):
                if param_shapes[i].value() != x_shapes[batch_ndims + i].value():
                    raise RuntimeError(
                        f"Layernorm {param_name} shape is not compatible with input shape. "
                        f"{param_name} shape: {param_shapes}, input shape: {x_shapes}"
                    )

        _check_param_shapes(x_shapes, gamma_shapes, "gamma")
        _check_param_shapes(x_shapes, beta_shapes, "beta")
        _check_param_shapes(x_shapes, normalized_shape, "normalized")

    @staticmethod
    def get_input_shapes(x, gamma, beta) -> List[List[Union[IntVar, IntImm]]]:
        """
        Return a list of shapes for x, gamma and beta, where gamma_shape and
        beta_shape may be None if gamma and beta are None, respectively.
        """
        x_shape = x._attrs["shape"]
        # gamma and beta can be None.
        gamma_shape = None
        if gamma is not None:
            gamma_shape = gamma._attrs["shape"]
        beta_shape = None
        if beta is not None:
            beta_shape = beta._attrs["shape"]
        return [x_shape, gamma_shape, beta_shape]

    def _sanity_check(self, x, gamma, beta):
        normalized_shape = self._attrs["normalized_shape"]

        # size() op can introduce up to 1 more input per normalized dim
        input_len = len(self._attrs["inputs"])
        max_input_len = 3 + len(normalized_shape)
        if input_len < 1 or input_len > max_input_len:
            raise NotImplementedError(
                f"Expect 1 ~ {max_input_len} inputs for Layernorm. Actual #inputs: {input_len}"
            )
        (x_shape, gamma_shape, beta_shape) = layernorm.get_input_shapes(x, gamma, beta)

        expected_dtype = x.dtype()
        for param, name in ((gamma, "gamma"), (beta, "beta")):
            if param is not None and param.dtype() != expected_dtype:
                raise NotImplementedError(
                    f"Layernorm doesn't support type promotions; expected {expected_dtype} but got {name} with dtype {param.dtype()}"
                )

        layernorm.check_shapes(x_shape, gamma_shape, beta_shape, normalized_shape)

    def _infer_shapes(self, x: Tensor):
        """Infer shapes for layernorm."""

        return x._attrs["shape"]

    def __call__(
        self,
        x: Tensor,
        gamma: Tensor = None,
        beta: Tensor = None,
        normalized_shape: List[Any] = None,
        eps: float = 1e-5,
    ) -> Tensor:
        inputs = [x]
        self._attrs["gamma_constant"] = "1.0"
        self._attrs["beta_constant"] = "0.0"
        if gamma is not None:
            self._attrs["gamma_constant"] = None
            inputs.append(gamma)
        if beta is not None:
            self._attrs["beta_constant"] = None
            inputs.append(beta)
        if normalized_shape is not None:
            new_norm_shape = shape_utils.convert_shape_to_IntVar(normalized_shape)
            # Only add source of dynamic dim to inputs
            for old_shape, new_shape in zip(normalized_shape, new_norm_shape):
                if not isinstance(new_shape, IntImm):
                    inputs.append(old_shape)
            self._attrs["normalized_shape"] = new_norm_shape
        else:
            self._attrs["normalized_shape"] = self._attrs["default_normalized_shape"]
        assert isinstance(eps, float), f"eps must be float, instead it is {type(eps)}"
        self._attrs["eps"] = eps
        self._attrs["inputs"] = inputs
        self._attrs["input_accessors"] = [TensorAccessor(x)]
        self._sanity_check(x, gamma, beta)
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        self._attrs["exec_cond_template"] = EXEC_COND_TEMPLATE
        func = registry.get(func_key)
        return func(self._attrs)

    def _invert_exec_key(self, key):
        """Invert execution key to get input arguments as integers.

        Parameters
        ----------
        key : str
            Execution key

        Returns
        ----------
            List[int]
        """
        res = []
        for item in re.split(" == | && ", key):
            if item.isnumeric():
                res.append(int(item))
        return res

    def _gen_exec_key(self, name_value_mapping):
        """Generate execution key from the name value mapping.

        Parameters
        ----------
        name_value_mapping : Dict[str, Union[int, List[int]]
            Dict for name and value.

        Returns
        ----------
            str
        """
        key_strs = []
        for name, values in name_value_mapping.items():
            if len(values) == 1:
                key_strs.append(f"{name} == {values[0]}")
            elif len(values) > 1:
                key_strs.append(f"{name} >= {values[0]} && {name} <= {values[-1]}")
            else:
                raise RuntimeError(
                    "Softmax input has empty dim values: {}".format(values)
                )
        return " && ".join(key_strs)

    def _extract_exec_path(self, dynamic_profiling_strategy=DynamicProfileStrategy.MAX):
        """Extract execution key, i.e. input arguments for the profiler.

        Parameters
        ----------
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy. By default MAX is used, i.e. to profile
            a dynamic range, an upper bound will be used.
        """
        assert (
            len(self._attrs["normalized_shape"]) == 1
        ), "For profiling, normalized_shape must be 1D"

        m_max = 1
        m_min = 1
        for dim in self._attrs["inputs"][0]._attrs["shape"][:-1]:
            m_max *= max(dim._attrs["values"])
            m_min *= min(dim._attrs["values"])

        n = self._attrs["inputs"][0]._attrs["shape"][-1].value()

        shape_values_dict = {
            "M": [m_min, m_max],
            "N": [n],
        }

        self._attrs["exec_path"] = OrderedDict()
        if dynamic_profiling_strategy == DynamicProfileStrategy.MAX:
            max_values = {"M": [m_max], "N": [n]}

            exec_item = ExecItem(
                profiling_key=self._gen_exec_key(max_values),
                exec_cond=self._gen_exec_key(shape_values_dict),
                algo="",
            )
            self._attrs["exec_path"][exec_item.profiling_key] = exec_item
        elif dynamic_profiling_strategy == DynamicProfileStrategy.MIN:
            min_values = {"M": [m_min], "N": [n]}
            exec_item = ExecItem(
                profiling_key=self._gen_exec_key(min_values),
                exec_cond=self._gen_exec_key(shape_values_dict),
                algo="",
            )
            self._attrs["exec_path"][exec_item.profiling_key] = exec_item

    def _gen_profile_cmd(self, profiler_prefix, cfg, x_shape):
        """Generate profiler command.

        Parameters
        ----------
        profiler_prefix : str
            Directory to store profiler.
        cfg: str
            The filename generated for profiler.
        x_shape : List[int]
            Input shapes for the profiler.
        """
        exe_path = os.path.join(profiler_prefix, cfg)
        if not os.access(exe_path, os.X_OK):
            raise RuntimeError("Profiler %s is not executable" % exe_path)
        cmd = [exe_path]
        for shape in x_shape:
            cmd.append(shape)
        command = [str(x) for x in cmd]
        return command

    def _profile_single_workload(self, profiler_prefix, exec_key, devices):
        """Profile a single workload.

        Parameters
        ----------
        profiler_prefix : str
            Base dir to keep profiling source codes.
        exec_key: str
            Input arguments to profiler executables.
        devices: List[int]
            GPU device ids used for profiling.
        """
        target = backend.target.Target.current()
        # if in CI just choose minimal configs
        # workspace is a hack just provides 102400 Byte
        # query cache
        tmp_key = next(iter(self._attrs["op_instance"].keys()))
        tmp_op = self._attrs["op_instance"][tmp_key]
        exec_entry_sha1 = sha1(exec_key.encode("utf-8")).hexdigest()
        query = NormQueryEntry(
            dtype_in=tmp_op.In.value,
            dtype_acc=tmp_op.accumulator_type().value,
            dtype_out=tmp_op.Out.value,
            rank=tmp_op.Rank,
            op_type=self._attrs["op"],
            device=target._arch,
            exec_entry_sha1=exec_entry_sha1,
        )
        cache_value = target.query_profile_cache("normalization", query.__dict__)
        if cache_value is not None and not target.force_profile():
            _LOGGER.info("Load profiling result from cache.")
            return cache_value

        content = list(self._attrs["op_instance"].keys())
        runner = backend.profiler_runner.Runner(devices, self._attrs["name"])
        x_shape = self._invert_exec_key(exec_key)
        for cfg in content:
            command = self._gen_profile_cmd(profiler_prefix, cfg, x_shape)
            runner.push(cfg, command)

        runner.join()
        result = runner.pull()

        if len(result) == 0:
            raise RuntimeError(
                "Profile workload: " f"{exec_key}" " failed. " f"Results: {result}."
            )

        out = min(result, key=lambda x: x[1].duration)
        best_algo = out[0]
        workspace = out[1].workspace
        ## cache
        cache_record = NormRecordEntry(
            exec_entry=exec_key,
            exec_entry_sha1=exec_entry_sha1,
            dtype_in=tmp_op.In.value,
            dtype_acc=tmp_op.accumulator_type().value,
            dtype_out=tmp_op.Out.value,
            rank=tmp_op.Rank,
            op_type=self._attrs["op"],
            device=target._arch,
            algo=best_algo,
            workspace=workspace,
        )
        Target.current().insert_profile_cache("normalization", cache_record.__dict__)
        return (best_algo, workspace)

    def profile(
        self,
        workdir="./",
        devices=None,
        dynamic_profiling_strategy=DynamicProfileStrategy.MAX,
    ):
        """Selects the fastest kernel configurations.

        Parameters
        ----------
        workdir : str, optional
            Base dir to keep profiling source codes, by default "./"
        devices: list, optional
            Devices used for profiling, by default device 0 will be used.
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy. By default MAX is used, i.e. to profile
            a dynamic range, an upper bound will be used.
        """

        if devices is None:
            devices = [0]

        self._extract_exec_path(dynamic_profiling_strategy)

        workloads = list(self._attrs["exec_path"].keys())
        profiler_prefix = os.path.join(workdir, "profiler", self._attrs["op"])
        if "op_instance" not in self._attrs:
            target = backend.target.Target.current()
            # init candidate ops
            func_key = "{target}.{op}.config".format(
                target=target.name(), op=self._attrs["op"]
            )
            func = registry.get(func_key)
            func(self._attrs)

        for wkl in workloads:
            _LOGGER.info(
                "Profile: {name}: {wkl}".format(name=self._attrs["name"], wkl=wkl),
            )
            best_algo, workspace = self._profile_single_workload(
                profiler_prefix, wkl, devices
            )
            self._attrs["exec_path"][wkl].algo = best_algo
            self._attrs["workspace"] = workspace

    def gen_profiler(
        self,
        workdir: str = None,
        dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
    ) -> None:
        """Generator profiler. The profiler files are standalone executable for profiling.

        Parameters
        ----------
        workdir : str, optional
            Base dir to keep profiling source codes, by default "./"
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy, used to filter generated profiles at compile time.
            See also: :func:`~aitemplate.compiler.transform.profile.profile`
        """
        target = Target.current()
        # init candidate ops
        func_key = "{target}.{op}.config".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        func(self._attrs)
        func_key = "{target}.{op}.gen_profiler".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs, workdir)

    def _get_op_attributes(self):
        return {"normalized_shape": self._attrs["default_normalized_shape"]}
