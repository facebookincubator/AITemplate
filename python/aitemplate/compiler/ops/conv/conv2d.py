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
Base class for conv2d.
"""
import itertools
import logging
import os
import re
from collections import OrderedDict
from hashlib import sha1
from typing import Any, Dict, List

import jinja2

from .... import backend
from ....backend import registry
from ....backend.target import Target
from ....utils import alignment, environ, shape_utils
from ...base import DynamicProfileStrategy, IntImm, IntVar, Operator, Tensor
from .cache_entry import ConvQueryEntry, ConvRecordEntry
from .conv_common import (
    filter_op_instances,
    generate_profiler_sources,
    get_profiler_filename,
)

# pylint: disable=C0103,W0221,R1732,W0102,W1202,C0301,R1716


_LOGGER = logging.getLogger(__name__)

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}CI = {{x_dim3}};
{{indent}}{{dtype}}CO = {{w_dim0}};
{{indent}}{{dtype}}KH = {{w_dim1}};
{{indent}}{{dtype}}KW = {{w_dim2}};
{{indent}}{{dtype}}SH = {{stride}};
{{indent}}{{dtype}}SW = {{stride}};
{{indent}}{{dtype}}DH = {{dilate}};
{{indent}}{{dtype}}DW = {{dilate}};
{{indent}}{{dtype}}PH = {{pad}};
{{indent}}{{dtype}}PW = {{pad}};
{{indent}}{{dtype}}KHEff = (KH - 1) * DH + 1;
{{indent}}{{dtype}}KWEff = (KW - 1) * DW + 1;
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}HO = (HI + PH + PH - KHEff) {{div}} SH + 1;
{{indent}}{{dtype}}WO = (WI + PW + PW - KWEff) {{div}} SW + 1;
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = HO;
{{indent}}{{y_dim2}} = WO;
{{indent}}{{y_dim3}} = CO;
"""
)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
NI == {{x_dim0}} && HI == {{x_dim1}} && WI == {{x_dim2}} && CI == {{x_dim3}}
"""
)

EXEC_DYN_KEY_TEMPLATE = jinja2.Template(
    """
NI >= {{x_dim0_lb}} && NI <= {{x_dim0_ub}} && HI == {{x_dim1}} && WI == {{x_dim2}} && CI == {{x_dim3}}
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class conv2d(Operator):
    r"""
    Applies a 2D convolution on input with size (N, H, W, C_in), and produces output with size (N, H_out, W_out, C_out) where N is batch size, H, W are the height and width of the image in pixels, and C is the number of channels.

    In the simplest case, the output value of the layer with input size
    :math:`(N, H, W, C_{\text{in}})` and output :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    where :math:`\star` is the valid 2D `cross-correlation`_ operator.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`pad` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points.

    * :attr:`dilate` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`group` controls the number of blocked connections from input channels
      to output channels.

    Args:
        input: input tensor of shape :math:`(N , H , W, \text{in\_channels})`

        weight: filters of shape :math:`(\text{out\_channels} , K_h, K_w, \frac{\text{in\_channels}}{\text{groups}})`

    This operator uses "channels_last" data format. Below is an example and its equivalence in PyTorch:

    .. highlight:: python
    .. code-block:: python

        X = Tensor(shape=[N, H, W, C_in], dtype="float16", name="images", is_input=True)
        W = Tensor(shape=[C_out, K_h, K_w, C_in], dtype="float16", name="weight", is_input=True)
        OP = aitemplate.compiler.ops.conv2d(stride=1, pad=1, dilate=1)
        Y = OP(X, W)

    .. highlight:: python
    .. code-block:: python

        X_pt = NHWC2NCHW(X_ait)
        W_pt = NHWC2NCHW(W_ait)

        Y_pt = torch.nn.functional.conv2d(X_pt, W_pt)
        Y = NCHW2NHWC(Y_pt)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """Conv2d constructor.

        Parameters
        ----------
        stride : int
            Stride of the convolution
        pad : int
            Size of padding to add to the input
        dilate : int, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
           Number of blocked connections from input
            channels to output channels, by default 1
        """
        super().__init__()
        self._attrs["op"] = "conv2d"
        self._attrs["stride"] = stride
        self._attrs["pad"] = pad
        self._attrs["dilate"] = dilate
        self._attrs["group"] = group
        self._attrs["has_profiler"] = True
        self._attrs["epilogue_alignment"] = 1
        self._attrs["epilogue"] = "LinearCombination"
        self._attrs["workspace"] = 0
        self._attrs["split_k"] = None
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE
        self.exec_key_template = EXEC_KEY_TEMPLATE
        self.exec_dyn_key_template = EXEC_DYN_KEY_TEMPLATE
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _infer_shape(self, x: List[int], w: List[int]) -> List[int]:
        if x[3] != w[3] * self._attrs["group"]:
            raise RuntimeError("X/W Shape mismatch for conv2d")
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            stride=self._attrs["stride"],
            pad=self._attrs["pad"],
            dilate=self._attrs["dilate"],
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
            w_dim0=w[0],
            w_dim1=w[1],
            w_dim2=w[2],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor, w: Tensor) -> List[int]:
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        w_shape = [var._attrs["values"][0] for var in w._attrs["shape"]]
        self._attrs["CO"] = w_shape[0]
        self._attrs["KH"] = w_shape[1]
        self._attrs["KW"] = w_shape[2]
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape, w_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            x._attrs["shape"][0],
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]
        return output_shape

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape: List[int]):
        return self.exec_key_template.render(
            x_dim0=shape[0], x_dim1=shape[1], x_dim2=shape[2], x_dim3=shape[3]
        ).replace("\n", "")

    def _gen_dyn_exec_key(self, dim0_lb, dim0_ub, dim1, dim2, dim3):
        return self.exec_dyn_key_template.render(
            x_dim0_lb=dim0_lb, x_dim0_ub=dim0_ub, x_dim1=dim1, x_dim2=dim2, x_dim3=dim3
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        self._attrs["exec_path"] = OrderedDict()
        for x_shape in x_shapes:
            key = self._gen_exec_key(x_shape)
            self._attrs["exec_path"][key] = ""

    def _signature(self):
        signature = "conv2d: K=[{kh}, {kw}], S=[{s}], P=[{p}], CO=[{co}]".format(
            kh=self._attrs["KH"],
            kw=self._attrs["KW"],
            s=self._attrs["stride"],
            p=self._attrs["pad"],
            co=self._attrs["CO"],
        )
        return signature

    def _extract_epilogue_alignment(self, output_shape: List[IntVar]) -> None:
        epilogue_dim = output_shape[-1]
        if not isinstance(epilogue_dim, IntImm):
            raise RuntimeError("Conv output last dimension must be static!")
        self._attrs["epilogue_alignment"] = alignment.find_max_alignment(
            number=epilogue_dim._attrs["values"][0],
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )

    def __call__(self, x: Tensor, w: Tensor) -> List[Tensor]:
        """Call conv2d with tensors x, w

        Parameters
        ----------
        x : Tensor
            in shape (N, H, W, C_in)
        w : Tensor
            in shape (C_out, K_h, K_w, C_in)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, H_out, W_out, C_out)
        """
        self._attrs["inputs"] = [x, w]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self) -> Dict[str, Any]:
        target_attrs = ["dilate", "group", "pad", "stride"]
        attr = {}

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr

    def _should_build_profiler(self) -> bool:
        """
        Check if we should build profilers. If we have a cached
        entry for this conv instance, we update this conv op's
        relevant attributes with the cached result and return False.
        """
        force_cache = environ.force_profiler_cache()
        if self._has_dynamic_input_dims():
            if force_cache:
                raise RuntimeError(
                    "We cannot force to use the cache as dynamic dims require "
                    "us to generate and build the profilers"
                )
            # If there are dynamic dims, we'll have to generate and build the
            # profilers, as the binaries will be needed for dynamic profiling.
            return True
        # We are forced to use the cache so we skip building profilers.
        if force_cache:
            return False

        target = backend.target.Target.current()
        workloads = list(self._attrs["exec_path"].keys())

        build_profiler = True
        # Now, let's query if all of our workloads have cache entries. If that
        # is the case, it is safely to skip generating and building profilers.
        if not target.use_dummy_profiling_results():
            tmp_key = next(iter(self._attrs["op_instance"].keys()))
            tmp_op = self._attrs["op_instance"][tmp_key]
            build_profiler = False
            for wkl in workloads:
                exec_entry_sha1 = sha1(wkl.encode("utf-8")).hexdigest()
                split_k = (
                    1 if self._attrs["split_k"] is None else self._attrs["split_k"]
                )
                query = ConvQueryEntry(
                    dtype_a=tmp_op.A.element.value,
                    dtype_b=tmp_op.B.element.value,
                    dtype_c=tmp_op.C.element.value,
                    dtype_acc=tmp_op.accumulator_type().value,
                    major_a=tmp_op.A.layout.value,
                    major_b=tmp_op.B.layout.value,
                    major_c=tmp_op.C.layout.value,
                    kh=self._attrs["KH"],
                    kw=self._attrs["KW"],
                    co=self._attrs["CO"],
                    stride=self._attrs["stride"],
                    pad=self._attrs["pad"],
                    dilate=self._attrs["dilate"],
                    op_type=self._attrs["op"],
                    device=target._arch,
                    epilogue=tmp_op.epilogue_functor.value,
                    split_k=split_k,
                    exec_entry_sha1=exec_entry_sha1,
                )
                cache_value = target.query_profile_cache("conv", query.__dict__)
                if cache_value is not None and not target.force_profile():
                    _LOGGER.info(
                        f'Load profiling result for {self._attrs["name"]} '
                        f"from cache: {cache_value}",
                    )
                    best_algo, workspace = cache_value
                    self._attrs["exec_path"][wkl] = best_algo
                    self._attrs["workspace"] = max(self._attrs["workspace"], workspace)
                else:
                    # cache miss - we will have to generate and build profilers
                    build_profiler = True
        return build_profiler

    def gen_profiler(
        self,
        workdir: str = None,
        dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
    ) -> None:
        """Profiler generator.

        Parameters
        ----------
        workdir : str, optional, by default None
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy, used to filter generated profiles at compile time.
            See also: :func:`~aitemplate.compiler.transform.profile.profile`
        """
        target = backend.target.Target.current()

        func_key = "{target}.{op}.config".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        func(self._attrs)

        if self._should_build_profiler():
            x_shapes = [
                self._invert_exec_key(exec_key) for exec_key in self._attrs["exec_path"]
            ]
            self._attrs["op_instance"] = filter_op_instances(
                func_attrs=self._attrs,
                x_shapes=x_shapes,
            )
            return generate_profiler_sources(
                func_attrs=self._attrs,
                op_class="conv",
                workdir=workdir,
                shape_template=self.shape_eval_template,
            )

    def _gen_profile_cmd(self, profiler_prefix, cfg, x_shape):
        exe_path = os.path.join(profiler_prefix, cfg)
        if not os.access(exe_path, os.X_OK):
            raise RuntimeError("Profiler %s is not executable" % exe_path)
        cmd = [exe_path]
        cmd.append(x_shape[0])
        cmd.append(x_shape[1])
        cmd.append(x_shape[2])
        cmd.append(x_shape[3])
        cmd.append(self._attrs["KH"])
        cmd.append(self._attrs["KW"])
        cmd.append(self._attrs["CO"])
        cmd.append(self._attrs["stride"])
        cmd.append(self._attrs["pad"])
        cmd.append(self._attrs["dilate"])
        cmd.append(self._attrs["group"])
        command = [str(x) for x in cmd]
        return command

    def _profile_single_workload(self, profiler_prefix, exec_key, devices, force_cache):
        target = backend.target.Target.current()
        # query cache
        tmp_key = next(iter(self._attrs["op_instance"].keys()))
        tmp_op = self._attrs["op_instance"][tmp_key]
        exec_entry_sha1 = sha1(exec_key.encode("utf-8")).hexdigest()
        split_k = 1 if self._attrs["split_k"] is None else self._attrs["split_k"]
        query = ConvQueryEntry(
            dtype_a=tmp_op.A.element.value,
            dtype_b=tmp_op.B.element.value,
            dtype_c=tmp_op.C.element.value,
            dtype_acc=tmp_op.accumulator_type().value,
            major_a=tmp_op.A.layout.value,
            major_b=tmp_op.B.layout.value,
            major_c=tmp_op.C.layout.value,
            kh=self._attrs["KH"],
            kw=self._attrs["KW"],
            co=self._attrs["CO"],
            stride=self._attrs["stride"],
            pad=self._attrs["pad"],
            dilate=self._attrs["dilate"],
            op_type=self._attrs["op"],
            device=target._arch,
            epilogue=tmp_op.epilogue_functor.value,
            split_k=split_k,
            exec_entry_sha1=exec_entry_sha1,
        )
        cache_value = target.query_profile_cache("conv", query.__dict__)
        if cache_value is not None and not target.force_profile():
            _LOGGER.info("Load profiling result from cache.")
            return cache_value
        if cache_value is None and force_cache:
            op_type = self._attrs["op"]
            raise RuntimeError(
                "force_cache is enabled but we could not find the following cache ",
                f"available on device {target._arch=}, {op_type=}, {exec_entry_sha1=}",
            )
        if target.use_dummy_profiling_results():
            op_type = self._attrs["op"]
            raise Exception(
                "This is a CI run but we could not find the following cache ",
                f"available on device {target._arch}\n",
                f"{op_type} {exec_entry_sha1}.\n",
                "Please adjust target.select_minimal_algo function.",
            )
        if target.name() == "rocm":
            runner = backend.profiler_runner.Runner(
                devices, self._attrs["name"], timeout=1800
            )
            op_type = self._attrs["op"]
            all_op_names = list(self._attrs["op_instance"].keys())
            for op_name in all_op_names:
                x_shape = self._invert_exec_key(exec_key)
                command = self._gen_profile_cmd(profiler_prefix, op_name, x_shape)
                runner.push(op_name, command)
        else:
            profiler_filename = get_profiler_filename(self._attrs, "conv")
            runner = backend.profiler_runner.Runner(
                devices, self._attrs["name"], timeout=180
            )
            x_shape = self._invert_exec_key(exec_key)
            command = self._gen_profile_cmd(profiler_prefix, profiler_filename, x_shape)
            runner.push(profiler_filename, command)
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
        cache_record = ConvRecordEntry(
            exec_entry=exec_key,
            exec_entry_sha1=exec_entry_sha1,
            dtype_a=tmp_op.A.element.value,
            dtype_b=tmp_op.B.element.value,
            dtype_c=tmp_op.C.element.value,
            dtype_acc=tmp_op.accumulator_type().value,
            major_a=tmp_op.A.layout.value,
            major_b=tmp_op.B.layout.value,
            major_c=tmp_op.C.layout.value,
            kh=self._attrs["KH"],
            kw=self._attrs["KW"],
            co=self._attrs["CO"],
            stride=self._attrs["stride"],
            pad=self._attrs["pad"],
            dilate=self._attrs["dilate"],
            op_type=self._attrs["op"],
            epilogue=tmp_op.epilogue_functor.value,
            device=target._arch,
            algo=best_algo,
            workspace=workspace,
            split_k=split_k,  # todo add into profile
        )
        Target.current().insert_profile_cache("conv", cache_record.__dict__)
        return (best_algo, workspace)

    def _has_dynamic_input_dims(self):
        for input_tensor in self._attrs["inputs"]:
            for dim in input_tensor._attrs["shape"]:
                if not isinstance(dim, IntImm):
                    return True
        return False

    def profile(
        self,
        workdir="./",
        devices=None,
        dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
    ):
        if devices is None:
            devices = [0]
        self._profile_static(workdir, devices)

        target = backend.target.Target.current()
        if target.use_dummy_profiling_results():
            return

        if self._has_dynamic_input_dims():
            if dynamic_profiling_strategy != DynamicProfileStrategy.HINTS:
                raise NotImplementedError(
                    "conv2d only supports HINTS dynamic profiling strategy for now! Current strategy: {}".format(
                        dynamic_profiling_strategy
                    )
                )
            self._profile_dynamic_dim(workdir)

    def _profile_static(self, workdir, devices):
        """Profiles with static shapes."""

        workloads = list(self._attrs["exec_path"].keys())
        profiler_prefix = os.path.join(workdir, "profiler", self._attrs["op"])
        target = backend.target.Target.current()
        if "op_instance" not in self._attrs:
            # init candidate ops
            func_key = "{target}.{op}.config".format(
                target=target.name(), op=self._attrs["op"]
            )
            func = registry.get(func_key)
            func(self._attrs, dtype=self._attrs["inputs"][0]._attrs["dtype"])

        force_cache = environ.force_profiler_cache()
        for wkl in workloads:
            _LOGGER.info(
                "Profile: {name}: {wkl}".format(name=self._attrs["name"], wkl=wkl),
            )
            # if in CI just choose minimal configs
            # workspace is a hack just provides 102400 Byte
            if target.use_dummy_profiling_results() and not force_cache:
                algo = target.select_minimal_algo(
                    list(self._attrs["op_instance"].keys())
                )
                _LOGGER.info(f"Select minimal algo {algo} for CI")
                self._attrs["exec_path"][wkl] = algo
                self._attrs["workspace"] = 102400
            elif self._attrs["exec_path"][wkl] == "":
                best_algo, workspace = self._profile_single_workload(
                    profiler_prefix, wkl, devices, force_cache
                )
                self._attrs["exec_path"][wkl] = best_algo
                self._attrs["workspace"] = max(self._attrs["workspace"], workspace)

    def _profile_dynamic_dim(self, workdir):
        """Profiles with dynamic shapes."""

        profiler_prefix = os.path.join(workdir, "profiler", self._attrs["op"])
        runner = backend.profiler_runner.Runner([0], self._attrs["name"])
        # extract dynamic dim from exec_path
        if len(self._attrs["exec_path"]) <= 1:
            return
        if len(set(self._attrs["exec_path"].values())) <= 1:
            # all exec paths point to the same algo
            return

        def _extract_dynamic_dim(exec_keys):
            _LOGGER.info("ONLY SUPPORT DYNAMIC BATCH (dim0)!")
            var_dims = [[], [], [], []]
            for key in exec_keys:
                dims = self._invert_exec_key(key)
                for i, v in enumerate(dims):
                    var_dims[i].append(v)
            return var_dims

        dims = _extract_dynamic_dim(self._attrs["exec_path"].keys())
        dim1 = dims[1][0]
        dim2 = dims[2][0]
        dim3 = dims[3][0]
        algos = list(self._attrs["exec_path"].values())
        # generate region
        regions = []  # lb, ub, lb_algos, ub_algos
        for i in range(len(dims[0]) - 1):
            regions.append([dims[0][i], dims[0][i + 1], algos[i], algos[i + 1]])
        # for each region,
        #   binary search to find cutting point
        #   generate new exec
        special_cases = OrderedDict()
        new_exec_paths = OrderedDict()
        for lb, ub, lb_algo, ub_algo in regions:
            mid = (lb + ub) // 2
            origin_lb = lb
            origin_ub = ub
            last_mid = mid
            while mid > lb and mid < ub:
                mid = (lb + ub) // 2
                mid_shape = [mid, dim1, dim2, dim3]
                _LOGGER.info(
                    "current: lb_algo: {lb_algo}, LB:{lb} MID:{mid} UB:{ub}".format(
                        lb_algo=lb_algo, lb=lb, mid=mid, ub=ub
                    ),
                )

                # run the profiler binary with all ops on the mid_shape
                # and fetch the results only for the lb_algo and ub_algo
                profiler_filename = get_profiler_filename(self._attrs, "conv")
                profiler_cmd = self._gen_profile_cmd(
                    profiler_prefix, profiler_filename, mid_shape
                )
                runner.push(
                    idx=profiler_filename,
                    cmd=profiler_cmd,
                    return_ops=[str(lb_algo), str(ub_algo)],
                )
                runner.join()
                result = runner.pull()
                result_dict = {res.op_config: res for res in result[0][1]}

                assert len(result_dict) >= 1
                # if there is only one result, assume ub algo failed.
                if len(result_dict) == 1:
                    assert str(ub_algo) not in result_dict
                    # last_lb = lb
                    lb = mid + 1
                # if there are two result, compare to decide new lb/ub
                else:
                    lb_time = result_dict[str(lb_algo)].duration
                    ub_time = result_dict[str(ub_algo)].duration
                    if lb_time < ub_time:
                        # lb algo can work with larger batch
                        # last_lb = lb
                        lb = mid + 1
                    else:
                        # ub algo can work with smaller batch
                        # last_ub = ub
                        ub = mid - 1
                last_mid = mid
                mid = (lb + ub) // 2
            lo_region_key = self._gen_dyn_exec_key(
                origin_lb, last_mid, dim1, dim2, dim3
            )
            up_region_key = self._gen_dyn_exec_key(
                last_mid, origin_ub, dim1, dim2, dim3
            )
            new_exec_paths[lo_region_key] = lb_algo
            new_exec_paths[up_region_key] = ub_algo
            # find special cases
            # This code is kept in case need fully tested dynamic code
            # So far I find binary search works well.
            # def _find_special_case(lb, ub, algo):
            #     for i in range(lb + 1, ub + 1):
            #         x_shape = [i, dim1, dim2, dim3]
            #         cmd = self._gen_profile_cmd(profiler_prefix, str(algo), x_shape)
            #         runner.push(0, cmd)
            #         runner.join()
            #         out = runner.pull()
            #         if len(out) == 0:
            #             _LOGGER.info("Find specail case: batch=%d" % i)
            #             algo = self._profile_single_workload(profiler_prefix, x_shape, [0])
            #             special_cases[self._gen_exec_key(x_shape)] = algo

            # _LOGGER.info(
            #     "Searching for specail cases between [{lb}, {ub}]".format(lb=origin_lb,
            #         ub=last_mid))
            # _find_special_case(origin_lb, last_mid, lb_algo)
            # _LOGGER.info(
            #     "Searching for specail cases between [{lb}, {ub}]".format(lb=last_mid + 1,
            #         ub=origin_ub))
            # _find_special_case(last_mid, origin_ub, ub_algo)
        special_cases.update(new_exec_paths)
        self._attrs["exec_path"] = special_cases

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            self.exec_cond_template,
            self.shape_eval_template,
            self.shape_save_template,
        )
