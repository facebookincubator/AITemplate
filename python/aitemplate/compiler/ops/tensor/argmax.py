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
Argmax.
"""
import itertools
import logging
import os
import re
from collections import OrderedDict
from typing import List

import jinja2
import numpy as np

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import Operator, Tensor
from aitemplate.utils import shape_utils

# pylint: disable=C0103,W0221,W0102,W0223


_LOGGER = logging.getLogger(__name__)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
instance_size == {{x_dim0}} &&  instance_num == {{x_dim1}}
"""
)


class argmax(Operator):
    """
    Returns the indices of the maximum value of all elements across a dimension in the input tensor. If there are multiple maximal values then the indices of the first maximal value are returned.

    Args:
        input (Tensor): the source tensor
        dim (int): optional, the dimension to reduce. Default: 0

    Returns:
        Tensor: a long tensor that contains the indices of the maximum values
    """

    def __init__(self, dim=0) -> None:
        """initialize the op"""
        super().__init__()
        self._attrs["op"] = "argmax"
        self._attrs["has_profiler"] = False
        self._attrs["dim"] = dim
        self._attrs["has_profiler"] = True
        self._attrs["workspace"] = 0
        self.exec_key_template = EXEC_KEY_TEMPLATE

    def _infer_shape(self, x: List[int]):
        """Infer the output shape"""
        output = list(x)[:-1]
        return output

    def _infer_shapes(self, x: Tensor):
        """Infer the output shape"""
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = []
        for idx in range(len(y_shapes[0])):
            output_shape.append(
                shape_utils.gen_int_var(values=unique([d[idx] for d in y_shapes]))
            )
        return output_shape

    def __call__(self, x: Tensor) -> Tensor:
        """call the op

        Parameters
        ----------
        x : Tensor
            input tensor
        Returns
        ----------
            Tensor
        """
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        self._extract_exec_path(x)
        output = Tensor(output_shape, src_ops={self}, dtype="int64")
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """call backend function"""
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_profiler".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs, workdir)

    def _gen_exec_key(self, shape: List[int]):
        """rending the shape info"""
        elem_cnt = np.prod(shape)
        instance_size = shape[-1]
        instance_num = elem_cnt // instance_size
        return self.exec_key_template.render(
            x_dim0=instance_size, x_dim1=instance_num
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        self._attrs["exec_path"] = OrderedDict()
        for x_shape in x_shapes:
            key = self._gen_exec_key(x_shape)
            self._attrs["exec_path"][key] = ""

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_profile_cmd(self, profiler_prefix, cfg, x_shape):
        exe_path = os.path.join(profiler_prefix, cfg)
        if not os.access(exe_path, os.X_OK):
            raise RuntimeError("Profiler %s is not executable" % exe_path)
        cmd = [exe_path]
        cmd.append(x_shape[0])
        cmd.append(x_shape[1])
        command = [str(x) for x in cmd]
        _LOGGER.info("profiling cmd: {}".format(command))
        return command

    def _profile_single_workload(self, profiler_prefix, exec_key, devices):
        runner = backend.profiler_runner.Runner(devices, self._attrs["name"])
        cfg = self._attrs["op"]
        x_shape = self._invert_exec_key(exec_key)
        command = self._gen_profile_cmd(profiler_prefix, cfg, x_shape)
        runner.push(cfg, command)
        runner.join()
        result = runner.pull()

        if len(result) == 0:
            raise RuntimeError(
                "Profile workload: " f"{exec_key}" " failed. " f"Results: {result}."
            )

        out = min(result, key=lambda x: x[1].duration)
        workspace = out[1].workspace
        return workspace

    def profile(
        self,
        workdir="./",
        devices=None,
        dynamic_profiling_strategy=None,
    ):
        """Get the Argmax Op workspace
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

        workloads = list(self._attrs["exec_path"].keys())
        profiler_prefix = os.path.join(workdir, "profiler", self._attrs["op"])

        for wkl in workloads:
            _LOGGER.info(
                "Profile: {name}: {wkl}".format(name=self._attrs["name"], wkl=wkl),
            )
            workspace = self._profile_single_workload(profiler_prefix, wkl, devices)
            self._attrs["workspace"] = max(self._attrs["workspace"], workspace)
