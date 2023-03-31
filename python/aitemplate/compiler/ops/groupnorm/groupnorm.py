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
Operator definition for groupnorm.
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

from aitemplate.testing import detect_target

# pylint: disable=C0103,W0221,W0102,W0223


_LOGGER = logging.getLogger(__name__)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class group_norm(Operator):
    """Standalone group norm op.
    The grouped dim must be the last dim of the input tensor.
    """

    def __init__(self, num_groups: int, num_channels: int) -> None:
        super().__init__()
        self._attrs["op"] = "groupnorm"
        self._attrs["num_groups"] = num_groups
        self._attrs["has_profiler"] = False
        if detect_target().name() == "rocm":
            self._attrs["has_profiler"] = True
        self._attrs["num_channels"] = num_channels
        self._attrs["workspace"] = 0

    @staticmethod
    def check_shapes(x_shapes, gamma_shapes, beta_shapes, num_groups):
        # check last dim can be divided by num_groups
        # minimal group: 8
        if gamma_shapes is not None and beta_shapes is not None:
            if len(gamma_shapes) != len(beta_shapes):
                raise RuntimeError(
                    f"Gamma and beta must have the same number of dimensions, but got {len(gamma_shapes)} and {len(beta_shapes)}"
                )
            if x_shapes[-1].value() != gamma_shapes[0].value():
                raise RuntimeError(
                    f"Input last dim {x_shapes[-1]} must be equal to gamma dim {gamma_shapes[0]}"
                )
        if x_shapes[-1].value() % num_groups != 0:
            raise RuntimeError(
                f"Channel dim {gamma_shapes[0]} must be divisible by num_groups {num_groups}"
            )
        return

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
        (x_shape, gamma_shape, beta_shape) = group_norm.get_input_shapes(x, gamma, beta)
        group_norm.check_shapes(
            x_shape, gamma_shape, beta_shape, self._attrs["num_groups"]
        )

    def _infer_shapes(self, x: Tensor):
        """Infer shapes for groupnorm."""

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

        assert isinstance(eps, float), f"eps must be float, instead it is {type(eps)}"
        self._attrs["eps"] = eps
        self._attrs["inputs"] = inputs

        self._sanity_check(x, gamma, beta)
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x.dtype())

        batch_size = output_shape[0]._attrs["values"][-1]
        self._attrs["workspace"] = 8 * batch_size * self._attrs["num_groups"]
        self._attrs["outputs"] = [output]
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
            Dict[str, int]
        """
        vals = []
        key_strs = []
        for item in re.split(" == | && ", key):
            if item.isnumeric():
                vals.append(int(item))
            else:
                key_strs.append(item.strip())
        assert len(vals) == len(
            key_strs
        ), f"expected len(vals) == len(key_strs), but got {len(vals)}, {len(key_strs)}"
        return dict(zip(key_strs, vals))

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
                raise RuntimeError(f"Group norm input has empty dim values: {values}")
        return " && ".join(key_strs)

    def _gen_profile_cmd(self, profiler_prefix, cfg, x_shape_dict):
        """Generate profiler command.

        Parameters
        ----------
        profiler_prefix : str
            Directory to store profiler.
        cfg: str
            The filename generated for profiler.
        x_shape_dict : List[str, int]
            Input shapes for the profiler.
        """
        exe_path = os.path.join(profiler_prefix, cfg)
        if not os.access(exe_path, os.X_OK):
            raise RuntimeError("Profiler %s is not executable" % exe_path)
        cmd = [exe_path]
        x_shape = ["N", "H", "W", "G", "C"]
        for shape in x_shape:
            cmd.append(x_shape_dict[shape])
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
        x_shape_dict = self._invert_exec_key(exec_key)
        for cfg in content:
            command = self._gen_profile_cmd(profiler_prefix, cfg, x_shape_dict)
            runner.push(cfg, command)

        runner.join()
        result = runner.pull()

        if len(result) == 0:
            raise RuntimeError(
                "Profile workload: " f"{self._attrs['op']}" f"{exec_key}" " failed. " f"Results: {result}."
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

    def _extract_exec_path(self, dynamic_profiling_strategy=DynamicProfileStrategy.MAX):
        """Extract execution key, i.e. input arguments for the profiler.

        Parameters
        ----------
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy. By default MAX is used, i.e. to profile
            a dynamic range, an upper bound will be used.
        """
        n_dim = self._attrs["inputs"][0]._attrs["shape"][0]
        n_max = max(n_dim._attrs["values"])
        n_min = min(n_dim._attrs["values"])

        h_dim = self._attrs["inputs"][0]._attrs["shape"][1]
        assert isinstance(h_dim, IntImm), "groupnorm requires h_dim to be static"
        w_dim = self._attrs["inputs"][0]._attrs["shape"][2]
        assert isinstance(w_dim, IntImm), "groupnorm requires w_dim to be static"
        c_dim = self._attrs["inputs"][0]._attrs["shape"][3]
        assert isinstance(c_dim, IntImm), "groupnorm requires c_dim to be static"

        # N, H, W, G, C
        shape_values_dict = {
            "N": [n_min, n_max],
            "H": [h_dim.value()],
            "W": [w_dim.value()],
            "G": [self._attrs["num_groups"]],
            "C": [c_dim.value()],
        }

        self._attrs["exec_path"] = OrderedDict()
        if dynamic_profiling_strategy == DynamicProfileStrategy.MAX:
            max_values = {
                name: [max(shape_values)]
                for name, shape_values in shape_values_dict.items()
            }
            exec_item = ExecItem(
                profiling_key=self._gen_exec_key(max_values),
                exec_cond=self._gen_exec_key(shape_values_dict),
                algo="",
            )
            self._attrs["exec_path"][exec_item.profiling_key] = exec_item
        elif dynamic_profiling_strategy == DynamicProfileStrategy.MIN:
            min_values = {
                name: [min(shape_values)]
                for name, shape_values in shape_values_dict.items()
            }
            exec_item = ExecItem(
                profiling_key=self._gen_exec_key(min_values),
                exec_cond=self._gen_exec_key(shape_values_dict),
                algo="",
            )
            self._attrs["exec_path"][exec_item.profiling_key] = exec_item

    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"] + [f"num_groups={self._attrs['num_groups']}"]

    def _get_op_attributes(self):
        return {
            "num_groups": self._attrs["num_groups"],
            "num_channels": self._attrs["num_channels"],
        }
