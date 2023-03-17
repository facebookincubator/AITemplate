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
Topk.
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
from aitemplate.compiler.base import IntImm, IntVar, Operator, Tensor

# pylint: disable=C0103,W0221,W0102,W0223


_LOGGER = logging.getLogger(__name__)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
elem_cnt == {{x_dim0}} &&  instance_size == {{x_dim1}} &&  instance_num == {{x_dim2}}
"""
)


class topk(Operator):
    """Returns the k largest elements of the given input tensor along its last dimension.

    * :attr:`k` the k in "top-k".

    Args:
        x (Tensor) : the input tensor

    Return:
        Tensor : the output tensor with last dimension being `k`.

    Example:

        .. highlight:: python
        .. code-block:: python

            X = Tensor(shape=[2, 800], name="X", is_input=True)
            Y = ops.topk(k=300)(X)
            y_shape = [d._attrs["values"][0] for d in Y.shape()]
            print(y_shape)

            Outs:
            [2, 300]
    """

    def __init__(self, k) -> None:
        super().__init__()
        self._attrs["op"] = "topk"
        self._attrs["has_profiler"] = True
        self._attrs["topK"] = k
        self._attrs["workspace"] = 0
        self.exec_key_template = EXEC_KEY_TEMPLATE

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for topK."""

        output_shape = list(x._attrs["shape"])
        output_shape[-1] = IntImm(self._attrs["topK"])
        return output_shape

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        self._extract_exec_path(x)
        output = Tensor(output_shape, src_ops={self}, dtype="int64")
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {"k": self._attrs["topK"]}

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
        """Profile TopK to get workspace
        Parameters
        ----------
        workdir : str, optional
            [description], by default None
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy, used to filter generated profiles at compile time.
            See also: :func:`~aitemplate.compiler.transform.profile.profile`
        """
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
            x_dim0=elem_cnt, x_dim1=instance_size, x_dim2=instance_num
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
        cmd.append(x_shape[2])
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
        """Get the TopK Op workspace

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
