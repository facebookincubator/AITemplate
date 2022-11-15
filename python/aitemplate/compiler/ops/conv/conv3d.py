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
Base class for conv3d.
"""
import itertools
import os
import re
from collections import OrderedDict
from hashlib import sha1
from operator import itemgetter
from typing import Any, Dict, List

import jinja2

from .... import backend
from ....backend import registry
from ....backend.target import Target
from ....utils import logger, shape_utils
from ...base import DynamicProfileStrategy, IntImm, IntVar, Operator, Tensor
from .cache_entry import Conv3dQueryEntry, Conv3dRecordEntry

# pylint: disable=C0103,W0221,R1732,W0102,W1202,C0301,R1716


SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}DI = {{x_dim1}};
{{indent}}{{dtype}}HI = {{x_dim2}};
{{indent}}{{dtype}}WI = {{x_dim3}};
{{indent}}{{dtype}}CI = {{x_dim4}};
{{indent}}{{dtype}}CO = {{w_dim0}};
{{indent}}{{dtype}}KD = {{w_dim1}};
{{indent}}{{dtype}}KH = {{w_dim2}};
{{indent}}{{dtype}}KW = {{w_dim3}};
{{indent}}{{dtype}}SD = {{stride_d}};
{{indent}}{{dtype}}SH = {{stride_h}};
{{indent}}{{dtype}}SW = {{stride_w}};
{{indent}}{{dtype}}DD = {{dilate_d}};
{{indent}}{{dtype}}DH = {{dilate_h}};
{{indent}}{{dtype}}DW = {{dilate_w}};
{{indent}}{{dtype}}PD = {{pad_d}};
{{indent}}{{dtype}}PH = {{pad_h}};
{{indent}}{{dtype}}PW = {{pad_w}};
{{indent}}{{dtype}}KDEff = (KD - 1) * DD + 1;
{{indent}}{{dtype}}KHEff = (KH - 1) * DH + 1;
{{indent}}{{dtype}}KWEff = (KW - 1) * DW + 1;
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}DO = (DI + PD + PD - KDEff) {{div}} SD + 1;
{{indent}}{{dtype}}HO = (HI + PH + PH - KHEff) {{div}} SH + 1;
{{indent}}{{dtype}}WO = (WI + PW + PW - KWEff) {{div}} SW + 1;
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = DO;
{{indent}}{{y_dim2}} = HO;
{{indent}}{{y_dim3}} = WO;
{{indent}}{{y_dim4}} = CO;
"""
)

EXEC_KEY_TEMPLATE = jinja2.Template(
    """
NI == {{x_dim0}} && DI == {{x_dim1}} && HI == {{x_dim2}} && WI == {{x_dim3}} && CI == {{x_dim4}}
"""
)

EXEC_DYN_KEY_TEMPLATE = jinja2.Template(
    """
NI >= {{x_dim0_lb}} && NI <= {{x_dim0_ub}} && DI == {{x_dim1}} && HI == {{x_dim2}} && WI == {{x_dim3}} && CI == {{x_dim4}}
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class conv3d(Operator):
    r"""conv3d"""

    def __init__(self, stride, pad, dilate=1, group=1) -> None:
        """Conv3d constructor.

        Parameters
        ----------
        stride : int or tuple
            Stride of the convolution
        pad : int or tuple
            Size of padding to add to the input
        dilate : int ot tuple, optional
            Size of spacing between kernel elements, by default 1
        group : int, optional
           Number of blocked connections from input
            channels to output channels, by default 1
        """
        super().__init__()
        self._attrs["op"] = "conv3d"
        self._attrs["stride"] = stride
        if isinstance(stride, int):
            self._attrs["stride"] = (stride, stride, stride)
        self._attrs["pad"] = pad
        if isinstance(pad, int):
            self._attrs["pad"] = (pad, pad, pad)
        self._attrs["dilate"] = dilate
        if isinstance(dilate, int):
            self._attrs["dilate"] = (dilate, dilate, dilate)
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
        if x[4] != w[4] * self._attrs["group"]:
            raise RuntimeError("X/W Shape mismatch for conv3d")
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            stride_d=self._attrs["stride"][0],
            stride_h=self._attrs["stride"][1],
            stride_w=self._attrs["stride"][2],
            pad_d=self._attrs["pad"][0],
            pad_h=self._attrs["pad"][1],
            pad_w=self._attrs["pad"][2],
            dilate_d=self._attrs["dilate"][0],
            dilate_h=self._attrs["dilate"][1],
            dilate_w=self._attrs["dilate"][2],
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
            x_dim4=x[4],
            w_dim0=w[0],
            w_dim1=w[1],
            w_dim2=w[2],
            w_dim3=w[3],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["DO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor, w: Tensor) -> List[int]:
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        w_shape = [var._attrs["values"][0] for var in w._attrs["shape"]]
        self._attrs["CO"] = w_shape[0]
        self._attrs["KD"] = w_shape[1]
        self._attrs["KH"] = w_shape[2]
        self._attrs["KW"] = w_shape[3]
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
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[4] for d in y_shapes])),
        ]
        return output_shape

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape: List[int]):
        return self.exec_key_template.render(
            x_dim0=shape[0],
            x_dim1=shape[1],
            x_dim2=shape[2],
            x_dim3=shape[3],
            x_dim4=shape[4],
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
        signature = "conv3d: K=[{kd}, {kh}, {kw}], S=[{sd}, {sh}, {sw}], P=[{pd}, {ph}, {pw}], CO=[{co}]".format(
            kd=self._attrs["KD"],
            kh=self._attrs["KH"],
            kw=self._attrs["KW"],
            sd=self._attrs["stride"][0],
            sh=self._attrs["stride"][1],
            sw=self._attrs["stride"][2],
            pd=self._attrs["pad"][0],
            ph=self._attrs["pad"][1],
            pw=self._attrs["pad"][2],
            co=self._attrs["CO"],
        )
        return signature

    def _extract_epilogue_alignment(self, output_shape: List[IntVar]) -> None:
        epilogue_dim = output_shape[-1]
        if not isinstance(epilogue_dim, IntImm):
            raise RuntimeError("Conv output last dimension must be static!")
        shape = epilogue_dim._attrs["values"][0]
        if shape % 8 == 0:
            self._attrs["epilogue_alignment"] = 8
        elif shape % 4 == 0:
            self._attrs["epilogue_alignment"] = 4
        elif shape % 2 == 0:
            self._attrs["epilogue_alignment"] = 2

    def __call__(self, x: Tensor, w: Tensor) -> List[Tensor]:
        """Call conv3d with tensors x, w

        Parameters
        ----------
        x : Tensor
            in shape (N, D, H, W, C_in)
        w : Tensor
            in shape (C_out, K_d, K_h, K_w, C_in)

        Returns
        -------
        List[Tensor]
            includes the output tensor in shape (N, D_out, H_out, W_out, C_out)
        """
        self._attrs["inputs"] = [x, w]
        self._set_depth()
        output_shape = self._infer_shapes(x, w)
        self._extract_exec_path(x)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self) -> Dict[str, Any]:
        target_attrs = ["dilate", "group", "pad", "stride"]
        attr = {}

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr

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
        func_key = "{target}.{op}.gen_profiler".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs, workdir, self.shape_eval_template)

    def _gen_profile_cmd(self, profiler_prefix, cfg, x_shape):
        exe_path = os.path.join(profiler_prefix, cfg)
        if not os.access(exe_path, os.X_OK):
            raise RuntimeError("Profiler %s is not executable" % exe_path)
        cmd = [exe_path]
        cmd.append(x_shape[0])
        cmd.append(x_shape[1])
        cmd.append(x_shape[2])
        cmd.append(x_shape[3])
        cmd.append(x_shape[4])
        cmd.append(self._attrs["KD"])
        cmd.append(self._attrs["KH"])
        cmd.append(self._attrs["KW"])
        cmd.append(self._attrs["CO"])
        cmd.append(self._attrs["stride"][0])
        cmd.append(self._attrs["stride"][1])
        cmd.append(self._attrs["stride"][2])
        cmd.append(self._attrs["pad"][0])
        cmd.append(self._attrs["pad"][1])
        cmd.append(self._attrs["pad"][2])
        cmd.append(self._attrs["dilate"][0])
        cmd.append(self._attrs["dilate"][1])
        cmd.append(self._attrs["dilate"][2])
        cmd.append(self._attrs["group"])
        command = [str(x) for x in cmd]
        return command

    def _profile_single_workload(self, profiler_prefix, exec_key, devices):
        target = backend.target.Target.current()
        # if in CI just choose minimal configs
        # workspace is a hack just provides 102400 Byte
        if target.use_dummy_profiling_results():
            algo = target.select_minimal_algo(list(self._attrs["op_instance"].keys()))
            logger.info(__name__, f"Select minimal algo {algo} for CI")
            return (algo, 102400)
        # query cache
        tmp_key = next(iter(self._attrs["op_instance"].keys()))
        tmp_op = self._attrs["op_instance"][tmp_key]
        exec_entry_sha1 = sha1(exec_key.encode("utf-8")).hexdigest()
        split_k = 1 if self._attrs["split_k"] is None else self._attrs["split_k"]
        query = Conv3dQueryEntry(
            dtype_a=tmp_op.A.element.value,
            dtype_b=tmp_op.B.element.value,
            dtype_c=tmp_op.C.element.value,
            dtype_acc=tmp_op.tile_description.math_instruction.element_accumulator.value,
            major_a=tmp_op.A.layout.value,
            major_b=tmp_op.B.layout.value,
            major_c=tmp_op.C.layout.value,
            kd=self._attrs["KD"],
            kh=self._attrs["KH"],
            kw=self._attrs["KW"],
            co=self._attrs["CO"],
            stride_d=self._attrs["stride"][0],
            stride_h=self._attrs["stride"][1],
            stride_w=self._attrs["stride"][2],
            pad_d=self._attrs["pad"][0],
            pad_h=self._attrs["pad"][1],
            pad_w=self._attrs["pad"][2],
            dilate_d=self._attrs["dilate"][0],
            dilate_h=self._attrs["dilate"][1],
            dilate_w=self._attrs["dilate"][2],
            op_type=self._attrs["op"],
            device=target._arch,
            epilogue=tmp_op.epilogue_functor.value,
            split_k=split_k,
            exec_entry_sha1=exec_entry_sha1,
        )
        cache_value = target.query_profile_cache("conv3d", query.__dict__)
        if cache_value is not None and not target.force_profile():
            logger.info(__name__, "Load profiling result from cache.")
            return cache_value
        if target.use_dummy_profiling_results():
            op_type = self._attrs["op"]
            raise Exception(
                "This is a CI run but we could not find the following cache ",
                f"available on device {target._arch}\n",
                f"{op_type} {exec_entry_sha1}.\n",
                "To bypass, you need to make it available in the db table.",
            )

        func_key = "{target}.{op}.filter".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        content = list(self._attrs["op_instance"].keys())
        runner = backend.profiler_runner.Runner(devices, self._attrs["name"])
        x_shape = self._invert_exec_key(exec_key)
        for cfg in content:
            if not func(cfg, self._attrs, x_shape):
                continue
            command = self._gen_profile_cmd(profiler_prefix, cfg, x_shape)
            runner.push(cfg, command)

        runner.join()
        result = runner.pull()
        if len(result) == 0:
            raise RuntimeError(
                "Profile workload: " f"{exec_key}" " failed. " f"Results: {result}."
            )
        out = min(result, key=itemgetter(1))
        best_algo = out[0]
        workspace = out[1].workspace
        ## cache
        cache_record = Conv3dRecordEntry(
            exec_entry=exec_key,
            exec_entry_sha1=exec_entry_sha1,
            dtype_a=tmp_op.A.element.value,
            dtype_b=tmp_op.B.element.value,
            dtype_c=tmp_op.C.element.value,
            dtype_acc=tmp_op.tile_description.math_instruction.element_accumulator.value,
            major_a=tmp_op.A.layout.value,
            major_b=tmp_op.B.layout.value,
            major_c=tmp_op.C.layout.value,
            kd=self._attrs["KD"],
            kh=self._attrs["KH"],
            kw=self._attrs["KW"],
            co=self._attrs["CO"],
            stride_d=self._attrs["stride"][0],
            stride_h=self._attrs["stride"][1],
            stride_w=self._attrs["stride"][2],
            pad_d=self._attrs["pad"][0],
            pad_h=self._attrs["pad"][1],
            pad_w=self._attrs["pad"][2],
            dilate_d=self._attrs["dilate"][0],
            dilate_h=self._attrs["dilate"][1],
            dilate_w=self._attrs["dilate"][2],
            op_type=self._attrs["op"],
            epilogue=tmp_op.epilogue_functor.value,
            device=target._arch,
            algo=best_algo,
            workspace=workspace,
            split_k=split_k,  # todo add into profile
        )
        Target.current().insert_profile_cache("conv3d", cache_record.__dict__)
        return (best_algo, workspace)

    def profile(
        self,
        workdir="./",
        devices=None,
        dynamic_profiling_strategy=DynamicProfileStrategy.HINTS,
    ):
        if devices is None:
            devices = [0]
        self._profile_static(workdir, devices)

        has_dynamic = False
        for input_tensor in self._attrs["inputs"]:
            for dim in input_tensor._attrs["shape"]:
                if not isinstance(dim, IntImm):
                    has_dynamic = True
                    break
        if has_dynamic:
            if dynamic_profiling_strategy != DynamicProfileStrategy.HINTS:
                raise NotImplementedError(
                    "conv3d only supports HINTS dynamic profiling strategy for now! Current strategy: {}".format(
                        dynamic_profiling_strategy
                    )
                )
            self._profile_dynamic_dim(workdir)

    def _profile_static(self, workdir, devices):
        """Profiles with static shapes."""

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
            logger.info(
                __name__,
                "Profile: {name}: {wkl}".format(name=self._attrs["name"], wkl=wkl),
            )
            best_algo, workspace = self._profile_single_workload(
                profiler_prefix, wkl, devices
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

        def _extract_dynamic_dim(exec_keys):
            logger.info(__name__, "ONLY SUPPORT DYNAMIC BATCH (dim0)!")
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
                logger.info(
                    __name__,
                    "current: lb_algo: {lb_algo}, LB:{lb} MID:{mid} UB:{ub}".format(
                        lb_algo=lb_algo, lb=lb, mid=mid, ub=ub
                    ),
                )

                mid_lb_algo_cmd = self._gen_profile_cmd(
                    profiler_prefix, str(lb_algo), mid_shape
                )
                mid_ub_algo_cmd = self._gen_profile_cmd(
                    profiler_prefix, str(ub_algo), mid_shape
                )
                runner.push(0, mid_lb_algo_cmd)
                runner.push(1, mid_ub_algo_cmd)
                runner.join()
                result = runner.pull()
                assert len(result) >= 1
                # if there is only one result, assume ub algo failed.
                if len(result) == 1:
                    assert result[0][0] == 0
                    # last_lb = lb
                    lb = mid + 1
                # if there are two result, compare to decide new lb/ub
                else:
                    lb_time = result[0][1]
                    ub_time = result[1][1]
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
            #             logger.info(self._attrs["name"], "Find specail case: batch=%d" % i)
            #             algo = self._profile_single_workload(profiler_prefix, x_shape, [0])
            #             special_cases[self._gen_exec_key(x_shape)] = algo

            # logger.info(self._attrs["name"],
            #     "Searching for specail cases between [{lb}, {ub}]".format(lb=origin_lb,
            #         ub=last_mid))
            # _find_special_case(origin_lb, last_mid, lb_algo)
            # logger.info(self._attrs["name"],
            #     "Searching for specail cases between [{lb}, {ub}]".format(lb=last_mid + 1,
            #         ub=origin_ub))
            # _find_special_case(last_mid, origin_ub, ub_algo)
        special_cases.update(new_exec_paths)
        self._attrs["exec_path"] = special_cases

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        op_name = self._attrs["op"]
        func_key = "{target}.{op}.gen_function".format(target=target.name(), op=op_name)
        func = registry.get(func_key)
        return func(
            self._attrs,
            self.exec_cond_template,
            self.shape_eval_template,
            self.shape_save_template,
        )
