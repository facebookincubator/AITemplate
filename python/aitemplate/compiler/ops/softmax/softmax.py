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
Softmax op implementation
"""
import logging
import os
import re
from collections import OrderedDict
from hashlib import sha1
from typing import Dict, List, Union

import jinja2

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.backend.target import Target
from aitemplate.compiler.base import (
    DynamicProfileStrategy,
    ExecItem,
    IntVar,
    Operator,
    Tensor,
)
from aitemplate.compiler.ops.softmax.cache_entry import NormQueryEntry, NormRecordEntry

from aitemplate.testing import detect_target

from aitemplate.utils.tensor_utils import wrap_dim


_LOGGER = logging.getLogger(__name__)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class softmax(Operator):
    r"""Applies the Softmax function to a 2D input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Args:
        input (Tensor [N, M]):
        dim (int): optional, a dimension along which Softmax will be computed (so every slice
        along dim will sum to 1). Default: None, in this case the input tensor will be treated as
        a 1-D tensor.

    Returns:
        Tensor: a Tensor of the same dimension and shape as the input with
        values in the range [0, 1].
    """

    def __init__(
        self,
    ) -> None:
        """initialize the op"""
        super().__init__()
        self._attrs["op"] = "softmax"
        self._attrs["has_profiler"] = False
        if detect_target().name() == "rocm":
            self._attrs["has_profiler"] = True

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infer output shape for the softmax op.

        Parameters
        ----------
        x : Tensor
            Input shape of softmax op.

        Returns
        ----------
            List[IntVar]
        """
        shapes = x._attrs["shape"]
        assert (
            len(shapes) >= 2
        ), f"softmax only supports input with rank >= 2, current rank: {len(shapes)}"
        return x._attrs["shape"]

    def _invert_exec_key(self, key: str):
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

    def _gen_exec_key(self, name_value_mapping: Dict[str, Union[int, List[int]]]):
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
        shape_values_dict = {
            var._attrs["name"]: var._attrs["values"]
            for var in self._attrs["inputs"][0]._attrs["shape"]
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

    def __call__(self, x: Tensor, dim: int = None) -> Tensor:
        """call the op

        Parameters
        ----------
        x : Tensor
            input tensor
        dim : int
            the dimension to be normalized.
            (default: None, in this case the input tensor will be treated as
             a 1-D tensor)

        Returns
        ----------
            Tensor
        """
        if dim is None:
            raise NotImplementedError(
                "flattening input tensor before normalization is not supported yet"
            )
        dim = wrap_dim(dim, x._rank())
        if dim != x._rank() - 1:
            raise NotImplementedError(
                f"softmax currently only supports dim=x._rank() - 1, dim={dim}, x._rank()={x._rank()}"
            )

        self._attrs["inputs"] = [x]
        self._attrs["dim"] = dim
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x.dtype())
        self._attrs["outputs"] = [output]
        return output

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

    def gen_function(self) -> str:
        """Generate function body.

        Returns
        -------
        str
            The rendered template of generated function body.
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        self._attrs["exec_cond_template"] = EXEC_COND_TEMPLATE
        func = registry.get(func_key)
        return func(self._attrs)
