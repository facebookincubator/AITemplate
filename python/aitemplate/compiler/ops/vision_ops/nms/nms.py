# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary]
"""
import itertools
import os
import re
from collections import OrderedDict
from typing import List

import jinja2

from ..... import backend
from .....backend import registry
from .....utils import logger, shape_utils
from ....base import Operator, Tensor

# pylint: disable=C0103,W0221,W0102,W0223

# TODO: change to column last
SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}CI = {{x_dim1}};
{{indent}}{{dtype}}HI = {{x_dim2}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}CO = {{nmsMaxOut}};
{{indent}}{{dtype}}HO = HI;
"""
)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
num_batch == {{x_dim0}} &&  num_rois == {{x_dim1}}
"""
)


class nms(Operator):
    """nms implementation

    Parameters
    ----------
    Operator : [type]
        [description]
    """

    def __init__(
        self, preNmsTop=2000, nmsMaxOut=200, iouThreshold=0.5, minBoxSize=0
    ) -> None:
        """initialize the op"""
        super().__init__()
        self._attrs["op"] = "nms"
        self._attrs["has_profiler"] = False
        self._attrs["preNmsTop"] = preNmsTop
        self._attrs["nmsMaxOut"] = nmsMaxOut
        self._attrs["iouThreshold"] = iouThreshold
        self._attrs["minBoxSize"] = minBoxSize
        self._attrs["has_profiler"] = True
        self._attrs["workspace"] = 0
        self.exec_key_template = EXEC_KEY_TEMPLATE
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE

    def _infer_shape(self, x: List[int], w: List[int]):
        """Infer the output shape"""
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            nmsMaxOut=self._attrs["nmsMaxOut"],
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204  # noqa: P204
        return [int(output["NO"]), int(output["CO"]), int(output["HO"])]

    def _infer_shapes(self, x: Tensor, w: Tensor):
        """Infer the output shape"""
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        w_shape = [var._attrs["values"][0] for var in w._attrs["shape"]]
        self._attrs["KH"] = w_shape[0]
        self._attrs["KW"] = w_shape[1]
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape, w_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
        ]
        return output_shape

    def __call__(self, x: Tensor, scores: Tensor) -> Tensor:
        """call the op

        Parameters
        ----------
        x : Tensor
            input tensor
        scores : Tensor
            score tensor for sorting
        Returns
        ----------
            Tensor
        """
        self._attrs["inputs"] = [x, scores]
        self._set_depth()
        output_shape = self._infer_shapes(x, scores)
        self._extract_exec_path(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def _gen_exec_key(self, shape):
        """rending the shape info"""
        return self.exec_key_template.render(
            x_dim0=shape[0],
            x_dim1=shape[1],
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        self._attrs["exec_path"] = OrderedDict()
        for x_shape in x_shapes:
            key = self._gen_exec_key(x_shape)
            self._attrs["exec_path"][key] = ""

    def gen_function(self) -> str:
        """call backend function"""
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_decl(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_decl".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_function_call(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function_call".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=None
    ) -> None:
        """Profile NMS to get workspace
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
        func(self._attrs, workdir)

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
        logger.info(__name__, "profiling cmd: {}".format(command))
        return command

    def _profile_single_workload(self, profiler_prefix, exec_key, devices):
        runner = backend.profiler_runner.Runner(devices, self._attrs["name"])
        cfg = self._attrs["op"]
        x_shape = self._invert_exec_key(exec_key)
        command = self._gen_profile_cmd(profiler_prefix, cfg, x_shape)
        runner.push(cfg, command)
        runner.join()
        result = runner.pull()

        out = sorted(result, key=lambda x: x[1])
        if len(out) == 0:
            raise RuntimeError(
                "Profile workload: " + "" + "failed. " "Results: {}.".format(result)
            )
        workspace = out[0][1].workspace
        return workspace

    def profile(
        self,
        workdir="./",
        devices=None,
        dynamic_profiling_strategy=None,
    ):
        """Get the NMS Op workspace
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
            logger.info(
                __name__,
                "Profile: {name}: {wkl}".format(name=self._attrs["name"], wkl=wkl),
            )
            workspace = self._profile_single_workload(profiler_prefix, wkl, devices)
            self._attrs["workspace"] = max(self._attrs["workspace"], workspace)
