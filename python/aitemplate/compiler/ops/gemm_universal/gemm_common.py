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
Common functions/classes for GEMM ops
"""
import itertools
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
from operator import itemgetter
from typing import Any, Dict, List, Union

import jinja2

from aitemplate.backend.profiler_runner import ProfileResult

from .... import backend
from ....backend import registry
from ....utils import alignment, environ
from ...base import DynamicProfileStrategy, ExecItem, IntImm, IntVar, Operator, Tensor
from ...dtype import is_same_dtype
from ...tensor_accessor import TensorAccessor
from .cache_entry import GemmQueryEntry, GemmRecordEntry

# pylint: disable=C0103,R1711,W0102,W0221,E1120


_LOGGER = logging.getLogger(__name__)


def split_k_result_getter(result):
    return result[1].duration


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class Source(Enum):
    INPUT = 1
    OUTPUT = 2


@dataclass
class DimInfo:
    """Class to record dimension info."""

    def __init__(
        self,
        source: Source,
        tensor_idx: int,
        dim_idx: Union[int, List[int]],
        placeholder: bool = False,
    ):
        """
        source:
            Source.INPUT or Source.OUTPUT
        tensor_idx:
            Depending on source, extract info from inputs[tensor_idx] or outputs[tensor_idx]
        dim_idx:
            Extract shape from inputs/outputs[tensor_idx][dim_idx]
        placeholder:
            If True, the diminfo might not be accurate in compile time, just a placeholder to be filled afterwards
            This is useful to handle issue such as broadcasting which B might not be exact.
        """
        self.source = source
        self.tensor_idx = tensor_idx
        if isinstance(dim_idx, int):
            dim_idx = [dim_idx]
        self.dim_idx = dim_idx
        self.placeholder = placeholder

    source: Source
    tensor_idx: int
    dim_idx: List[int]
    placeholder: bool


def extract_shape_from_accessor(func_attrs, source: Source, idx: int):
    if source == Source.INPUT:
        if "input_accessors" in func_attrs:
            return func_attrs["input_accessors"][idx].original_shapes
        return func_attrs["inputs"][idx].shape()
    elif source == Source.OUTPUT:
        if "output_accessors" in func_attrs:
            return func_attrs["output_accessors"][idx].original_shapes
        return func_attrs["outputs"][idx].shape()
    else:
        raise RuntimeError(f"Unknown source, got {source}")


def create_input_batch_diminfo(input_shapes, batch_dims, output_batch):
    """
    Create inputs' batch diminfo.
    Provided input_shapes and the corresponding batch_dims, this function
    returns a list of batch's DimInfo of the inputs.

    input_shapes:
        A list of input shapes.
    batch_dims:
        The batch dimension for the corresponding input_shapes.
        If length of corresponding input's shape is less than 2, neglected.
    output_batch:
        The batch size for output.
    """
    assert len(input_shapes) == len(batch_dims)

    batch_diminfo = []
    for idx, input_shape in enumerate(input_shapes):
        if len(input_shape) > 2:
            batch_diminfo.append(
                DimInfo(
                    Source.INPUT,
                    tensor_idx=idx,
                    dim_idx=batch_dims[idx],
                    placeholder=input_shape[batch_dims[idx]] != output_batch,
                )
            )
    return batch_diminfo


def group_gemm_inverse_key_func(key):
    m_pattern = re.compile(r"GROUP_\d+_M\s*==\s*(\d+)")
    all_m = re.findall(m_pattern, key)
    n_pattern = re.compile(r"GROUP_\d+_N\s*==\s*(\d+)")
    all_n = re.findall(n_pattern, key)
    k_pattern = re.compile(r"GROUP_\d+_K\s*==\s*(\d+)")
    all_k = re.findall(k_pattern, key)
    assert len(all_m) == len(all_n) == len(all_n)
    return (all_m, all_n, all_k)


def gemm_inverse_key_func(key):
    tmp = re.findall(r"(\d+)", key)
    return [int(x) for x in tmp]


def default_align_ab(a, b, dtype):
    ab = math.gcd(a, b)
    return alignment.find_max_alignment(ab, dtype)


def _to_list(elem):
    if isinstance(elem, tuple):
        return list(elem)
    else:
        return [elem]


class gemm(Operator):
    """Base gemm operators"""

    def __init__(
        self,
    ):
        super().__init__()
        self._attrs["op"] = "gemm"
        self._attrs["has_profiler"] = True
        self._attrs["f_ab_alignment"] = None
        self._attrs["epilogue_alignment"] = 1
        self._attrs["epilogue"] = "LinearCombination"
        self._attrs["workspace"] = 0
        self._attrs["split_k"] = 1
        self._attrs["num_sources"] = 0
        self._attrs["alpha"] = 1.0
        self._attrs["permute_shape"] = ""
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _extract_epilogue_alignment(
        self, output_shape: List[Any], dynamic_profiling_strategy=None
    ) -> None:
        epilogue_dim = output_shape[-1]
        if isinstance(epilogue_dim, int):
            shape = epilogue_dim
        elif not isinstance(epilogue_dim, IntImm):
            # The alignment inferred here will be set to 1 during codegen.
            if dynamic_profiling_strategy is None:
                return
            elif dynamic_profiling_strategy == DynamicProfileStrategy.MAX:
                shape = epilogue_dim.upper_bound()
            elif dynamic_profiling_strategy == DynamicProfileStrategy.MIN:
                shape = epilogue_dim.lower_bound()
            else:
                raise RuntimeError(
                    f"Unsupported dynamic profiling strategy: {dynamic_profiling_strategy}"
                )
        else:
            shape = epilogue_dim._attrs["values"][0]

        dtype = self._attrs["inputs"][0].dtype()
        self._attrs["epilogue_alignment"] = alignment.find_max_alignment(shape, dtype)
        return

    def _infer_shapes(self, a: Tensor, b: Tensor):
        raise NotImplementedError("_infer_shapes() is not implemented!")

    def _gen_exec_key(self, name_value_mapping):
        key_strs = []
        for name, values in name_value_mapping.items():
            if len(values) == 1:
                key_strs.append(f"{name} == {values[0]}")
            elif len(values) > 1:
                key_strs.append(f"{name} >= {values[0]} && {name} <= {values[-1]}")
            else:
                raise RuntimeError("Gemm input has empty dim values: {}".format(values))
        return " && ".join(key_strs)

    def _extract_dims(self, for_profiling: bool = False) -> Dict[str, List[DimInfo]]:
        """Extracts a mapping between dim names and a list of DimInfo.
        This function will be used in gemm shape inference, gemm padding graph
        transformation, gemm profiling, etc.

        All subclasses must implement this API.

        An example result from gemm_rcr:
        {
            "M": [
                DimInfo(source=INPUT, tensor_idx=0, dim_idx=0),
                DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=0),
            ],
            "K": [
                DimInfo(source=INPUT, tensor_idx=0, dim_idx=1),
                DimInfo(source=INPUT, tensor_idx=1, dim_idx=1),
            ],
            "N": [
                DimInfo(source=INPUT, tensor_idx=1, dim_idx=0),
                DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=1),
            ],
        }


        Parameters
        ----------
        for_profiling: bool
            Whether this function is used for generating profiling source codes.
            If yes, some DimInfo are simplified. e.g. For gemm, we treat all tensors
            as 2d.
        """

        raise NotImplementedError("extract_dims() is not implemented!")

    def _extract_exec_path(self, dynamic_profiling_strategy):
        """Extracts profiling keys and execution conditions for a given dynamic_profiling_strategy.
        This function fills in self._attrs["exec_path"].
        Keys are "exec_key"s, and are used for profiling.
        Values are ItemValues, where "profiling_key" fields are the same as the corresponding keys,
        "exec_cond" fields specify dynamic ranges, and "algo" fields are empty for now.

        e.g. for gemm_rrr, input1=[m, k], input2=[k, n]
        m = 1, k = 128, n = 256.
        self._attrs["exec_path"] = {
            "M==1 && K==128 && N==256" : ItemValue(
                profiling_key="M==1 && K==128 && N==256",
                exec_cond="M==1 && K==128 && N==256",
                algo="",
            )
        }

        e.g. for gemm_rrr, input1=[dynamic_m, k], input2=[k, n]
        dynamic_m >= 1 and dynamic_m <= 1024, dynamic_profiling_strategy = MAX,
        k = 128, n = 256.
        self._attrs["exec_path"] = {
            "M==1024 && K==128 && N==256" : ItemValue(
                profiling_key="M==1024 && K==128 && N==256",
                exec_cond="M>=1 && M<=1024 && K==128 && N==256",
                algo="",
            )
        }

        Parameters
        ----------
        dynamic_profiling_strategy : DynamicProfileStrategy
            See comments for DynamicProfileStrategy.
        """

        dim_info_dict: Dict[str, List[DimInfo]] = self._extract_dims()
        dim_dict: Dict[str, IntVar] = {}
        for name, dim_infos in dim_info_dict.items():
            dim_info = None
            for d in dim_infos:
                if d.placeholder:
                    continue

                if dim_info is None:
                    dim_info = d
                elif d.source == Source.INPUT:
                    # input should have priority.
                    dim_info = d
            assert dim_info is not None, f"Couldn't find valid dim info for dim {name}"

            tensor_list = (
                self._attrs["inputs"]
                if dim_info.source == Source.INPUT
                else self._attrs["outputs"]
            )
            if dim_info.source == Source.INPUT and "input_accessors" in self._attrs:
                dim_dict[name] = _to_list(
                    itemgetter(*(dim_info.dim_idx))(
                        self._attrs["input_accessors"][
                            dim_info.tensor_idx
                        ].original_shapes
                    )
                )
            elif dim_info.source == Source.OUTPUT and "output_accessors" in self._attrs:
                dim_dict[name] = _to_list(
                    itemgetter(*(dim_info.dim_idx))(
                        self._attrs["output_accessors"][
                            dim_info.tensor_idx
                        ].original_shapes
                    )
                )
            else:
                dim_dict[name] = _to_list(
                    itemgetter(*(dim_info.dim_idx))(
                        tensor_list[dim_info.tensor_idx]._attrs["shape"]
                    )
                )
        shape_values_dict = {}
        for name, dims in dim_dict.items():
            min_value = math.prod([dim.lower_bound() for dim in dims])
            max_value = math.prod([dim.upper_bound() for dim in dims])
            shape_values_dict[name] = sorted({min_value, max_value})

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
        else:
            raise NotImplementedError(
                "Gemm only supports MIN or MAX dynamic profiling! "
                "Current dynamic_profiling_strategy: {}".format(
                    dynamic_profiling_strategy
                )
            )

    def _get_profiler_filename(self):
        """
        generate a filename for a profiler that benchmarks multiple GEMM instances
        """
        target = backend.target.Target.current()

        op_type = self._attrs["op"]
        all_op_names = list(self._attrs["op_instance"].keys())
        encoded_str = sha1((";".join(all_op_names)).encode("utf-8")).hexdigest()

        if target.use_dummy_profiling_results():
            # we don't use cache
            return f"{op_type}_{encoded_str}"
        else:
            cache_ver = target.get_profile_cache_version("gemm")
            return f"{op_type}_{encoded_str}_{cache_ver}"

    def _should_build_profiler(
        self, workloads: List[str], new_op_instance: OrderedDict
    ):
        """
        Check if we should build profilers. If we have a cached
        entry for this gemm instance, we update this gemm op's
        relevant attributes with the cached result and return False.
        """
        # We are forced to use the cache so we skip building profilers.
        if environ.force_profiler_cache():
            return False
        target = backend.target.Target.current()

        build_profiler = True
        # Now, let's query if all of our workloads have cache entries. If that
        # is the case, it is safely to skip generating and building profilers.
        if not target.use_dummy_profiling_results():
            tmp_key = next(iter(new_op_instance.keys()))
            tmp_op = new_op_instance[tmp_key]
            build_profiler = False
            for wkl in workloads:
                exec_entry_sha1 = sha1(wkl.encode("utf-8")).hexdigest()
                query = GemmQueryEntry(
                    dtype_a=tmp_op.A.element.value,
                    dtype_b=tmp_op.B.element.value,
                    dtype_c=tmp_op.C.element.value,
                    dtype_acc=tmp_op.accumulator_type().value,
                    major_a=tmp_op.A.layout.value,
                    major_b=tmp_op.B.layout.value,
                    major_c=tmp_op.C.layout.value,
                    op_type=self._attrs["op"],
                    device=target._arch,
                    epilogue=tmp_op.epilogue_functor.value,
                    exec_entry_sha1=exec_entry_sha1,
                    pshape=self._attrs["permute_shape"],
                )
                cache_value = target.query_profile_cache("gemm", query.__dict__)
                if cache_value is not None and not target.force_profile():
                    _LOGGER.info(
                        f'Load profiling result for {self._attrs["name"]} '
                        f"from cache: {cache_value}",
                    )
                    best_algo, workspace, split_k = cache_value
                    self._attrs["exec_path"][wkl].algo = best_algo
                    self._attrs["workspace"] = max(self._attrs["workspace"], workspace)
                    self._attrs["split_k"] = split_k
                else:
                    # cache miss - we will have to generate and build profilers
                    build_profiler = True
        return build_profiler

    def gen_profiler(
        self, workdir: str = None, dynamic_profiling_strategy=DynamicProfileStrategy.MAX
    ) -> None:
        """Generate profilers for this gemm op.

        Parameters
        ----------
        workdir : str, optional
            Output dir of profilers, by default None
        dynamic_profiling_strategy: DynamicProfileStrategy, optional
            A dynamic profiling strategy, used to filter generated profiles at compile time.
            See also: :func:`~aitemplate.compiler.transform.profile.profile`
        """
        target = backend.target.Target.current()
        # init candidate ops
        func_key = "{target}.{op}.config".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        func(self._attrs, dtype=self._attrs["inputs"][0]._attrs["dtype"])

        # init exec path
        self._extract_exec_path(dynamic_profiling_strategy)
        # init compile-time filter
        workloads = list(self._attrs["exec_path"].keys())
        ab_alignments = sorted({self._get_ab_alignment(wkl) for wkl in workloads})
        assert 1 == len(
            ab_alignments
        ), f"ab_alignments should be the same among all workloads, got {ab_alignments=}"
        func_key = "{target}.{op}.filter".format(
            target=target.name(), op=self._attrs["op"]
        )

        # Update epilogue alignment here because it may be different depending on the profiling strategy.
        # Note that this alignment is only used in profiling and will be updated
        # during the final codegen.
        # gemm_permute ops have special output alignment rules, skip here.
        if "layout" not in self._attrs:
            output_shape = self._attrs["output_accessors"][0].original_shapes
            self._extract_epilogue_alignment(output_shape, dynamic_profiling_strategy)

        filter_func = registry.get(func_key)
        # run compile-time filter
        new_op_instance = OrderedDict(
            {
                k: v
                for k, v in self._attrs["op_instance"].items()
                if filter_func(k, self._attrs, ab_alignments[0])
            }
        )
        _LOGGER.debug(
            f"Filtered profiler kernels for {self._attrs['op']}: reduced the "
            f"number of generated kernels from {len(self._attrs['op_instance'])} "
            f"to {len(new_op_instance)}",
        )
        self._attrs["op_instance"] = new_op_instance

        build_profiler = self._should_build_profiler(workloads, new_op_instance)
        if build_profiler:
            # generate profiler
            func_key = "{target}.{op}.gen_profiler".format(
                target=target.name(), op=self._attrs["op"]
            )
            func = registry.get(func_key)
            if target.name() == "rocm":
                return func(
                    self._attrs,
                    workdir,
                    self._extract_dims(for_profiling=True),
                )
            profiler_filename = self._get_profiler_filename()
            _LOGGER.info(f"generating {profiler_filename=}")
            return func(
                self._attrs,
                workdir,
                profiler_filename,
                self._extract_dims(for_profiling=True),
            )

    def _gen_profile_cmd(
        self, profiler_prefix, profiler_filename, exec_key, fbuild_cmd
    ):
        exe_path = os.path.join(profiler_prefix, profiler_filename)
        if not os.access(exe_path, os.X_OK):
            raise RuntimeError("Profiler %s is not executable" % exe_path)
        cmd_args = fbuild_cmd(exec_key)
        cmd = [exe_path]
        # mnk
        cmd.extend(cmd_args)
        command = [str(x) for x in cmd]
        # profiling gemm/bmm_permute with layout and shape for ROCM
        if self._attrs.get("shape") is not None:
            if backend.target.Target.current().name() == "rocm":
                for x in self._attrs["shape"]:
                    command.append(str(x))
        return command

    def _split_k_search_space(self, M, N, K):
        """Get split_k search range = [1] by default"""
        space = [1]
        # skip split-k search for rocm
        if backend.target.Target.current().name() == "rocm":
            return set(space)
        factor = K // max(M, N)
        low_range = max(1, factor // 4)
        high_range = min(factor, 32)
        if low_range == 1:
            low_range += 1
        space += list(range(low_range, high_range, 2))
        _LOGGER.debug(
            f"profiling split-k for gemm instance M={M}, N={N}, K={K} in {set(space)}",
        )
        return set(space)

    def _get_ab_alignment(self, exec_key):
        if self._attrs["op"].startswith("group_gemm"):
            all_m, all_n, all_k = group_gemm_inverse_key_func(exec_key)
            all_ab_alignments = [
                self._attrs["f_ab_alignment"](int(m), int(n), int(k))
                for m, n, k in zip(all_m, all_n, all_k)
            ]
            ab_alignment = min(all_ab_alignments)
        else:
            # exec_key may contain batch dimension, which we don't care here
            m, n, k = gemm_inverse_key_func(exec_key)[-3:]
            ab_alignment = self._attrs["f_ab_alignment"](m, n, k)
            if not alignment.valid_alignment(
                ab_alignment, self._attrs["inputs"][0].dtype()
            ):
                raise RuntimeError(
                    f"A / B {ab_alignment=} is not valid! The last dimension of each input tensor needs to be divisible by 2."
                    f"m: {m}, n: {n}, k: {k}."
                )
        return ab_alignment

    def _profile_single_workload(
        self, profiler_prefix, exec_key, profiler_runner, force_cache
    ):
        """
        Schedule profilers for given profiler path and gemm shape (exec_key)
        or get the result from cache
        or use dummy result in CI
        """
        target = backend.target.Target.current()
        tmp_key = next(iter(self._attrs["op_instance"].keys()))
        tmp_op = self._attrs["op_instance"][tmp_key]
        exec_entry_sha1 = sha1(exec_key.encode("utf-8")).hexdigest()
        split_k = 1 if self._attrs["split_k"] is None else self._attrs["split_k"]
        # Because we call gen_profiler to generate and compile all profilers
        # before running any of them, we won't be able to update the exec_path
        # in gen_profiler even if two gemms have the same problem size (assume that
        # we don't have a cache entry for this problem size). Consequently,
        # we still need to query the cache here to ensure we won't re-profile
        # the second gemm with the same problem size. Note that if we already
        # have a cache entry for the problem size before gen_profiler, we will
        # setup exec_path correctly in gen_profiler, so we won't get here at all.
        query = GemmQueryEntry(
            dtype_a=tmp_op.A.element.value,
            dtype_b=tmp_op.B.element.value,
            dtype_c=tmp_op.C.element.value,
            dtype_acc=tmp_op.accumulator_type().value,
            major_a=tmp_op.A.layout.value,
            major_b=tmp_op.B.layout.value,
            major_c=tmp_op.C.layout.value,
            op_type=self._attrs["op"],
            device=target._arch,
            epilogue=tmp_op.epilogue_functor.value,
            exec_entry_sha1=exec_entry_sha1,
            pshape=self._attrs["permute_shape"],
        )
        cache_value = target.query_profile_cache("gemm", query.__dict__)
        if cache_value is not None and not target.force_profile():
            _LOGGER.debug(
                f'Load profiling result for {self._attrs["name"]} '
                f"from cache: {cache_value}",
            )
            self._attrs["exec_path"][exec_key].algo = cache_value[0]
            self._attrs["workspace"] = max(self._attrs["workspace"], cache_value[1])
            self._attrs["split_k"] = cache_value[2]
            return
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
            op_type = self._attrs["op"]
            all_op_names = list(self._attrs["op_instance"].keys())
            for op_name in all_op_names:

                def _gen_callback(split_k):
                    def process_result_callback(result, postprocessing_delegate):
                        postprocessing_delegate.add_instance(
                            (result, self._attrs, op_name, exec_key, split_k)
                        )

                    return process_result_callback

                command = self._gen_profile_cmd(profiler_prefix, op_name, exec_key)
                if self._attrs["op"].startswith("group_gemm") or self._attrs[
                    "op"
                ].startswith("bmm"):
                    profiler_runner.push(command, _gen_callback(split_k=1))
                else:
                    m, n, k = gemm_inverse_key_func(exec_key)[-3:]
                    if "split_k_hints" in self._attrs:
                        split_k_search_space = self._attrs["split_k_hints"]
                    else:
                        split_k_search_space = self._split_k_search_space(m, n, k)
                    for split_k in split_k_search_space:
                        gemm_command = command + [str(split_k)]
                        profiler_runner.push(gemm_command, _gen_callback(split_k))
        else:
            profiler_filename = self._get_profiler_filename()

            def _gen_callback(split_k):
                def process_result_callback(result, postprocessing_delegate):
                    postprocessing_delegate.add_instance(
                        (result, self._attrs, profiler_filename, exec_key, split_k)
                    )

                return process_result_callback

            command = self._gen_profile_cmd(
                profiler_prefix, profiler_filename, exec_key
            )

            if self._attrs["op"].startswith("group_gemm") or self._attrs[
                "op"
            ].startswith("bmm"):
                profiler_runner.push(command, _gen_callback(split_k=1))
            else:
                m, n, k = gemm_inverse_key_func(exec_key)[-3:]
                if "split_k_hints" in self._attrs:
                    split_k_search_space = self._attrs["split_k_hints"]
                else:
                    split_k_search_space = self._split_k_search_space(m, n, k)
                for split_k in split_k_search_space:
                    gemm_command = command + [str(split_k)]
                    profiler_runner.push(gemm_command, _gen_callback(split_k))

    def profile(
        self,
        profiler_runner,
        workdir="./",
    ):
        """Selects the fastest kernel configurations.

        Parameters
        ----------
        profiler_runner: ProfilerRunner
            Profiler runner to schedule async profiler jobs,
        workdir : str
            Base dir to keep profiling source codes, by default "./"running on separate GPU devices concurrently
        """

        workloads = list(self._attrs["exec_path"].keys())
        profiler_prefix = os.path.join(workdir, "profiler", self._attrs["op"])
        if "op_instance" not in self._attrs:
            target = backend.target.Target.current()
            # init candidate ops
            func_key = "{target}.{op}.config".format(
                target=target.name(),
                op=self._attrs["op"],
            )
            func = registry.get(func_key)
            func(self._attrs, dtype=self._attrs["inputs"][0]._attrs["dtype"])
        target = backend.target.Target.current()
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
                self._attrs["exec_path"][wkl].algo = algo
                self._attrs["workspace"] = 102400
            elif self._attrs["exec_path"][wkl].algo != "":
                # we have cached best algo
                return
            else:
                self._profile_single_workload(
                    profiler_prefix, wkl, profiler_runner, force_cache
                )

    def gen_function(self) -> str:
        """Generates the function code for the gemm op for the current target.

        Returns
        -------
        str
            C++ source code of the function
        """
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            self.exec_cond_template,
            self._extract_dims(),
        )

    def _signature(self) -> str:
        """Generate the unique signature of the gemm op.

        Returns
        -------
        str
            The unique signature of the gemm op.
        """
        op_name = self._attrs["op"] + ("split_" + str(self._attrs["split_k"]))
        signature = sha1(op_name.encode("utf-8")).hexdigest()
        return signature

    def _align_ab(self, a: Tensor, b: Tensor):
        return a, b

    def _sanity_check(self, a: Tensor, b: Tensor):
        a_shapes = a._attrs["shape"]
        if len(a_shapes) < 2:
            raise RuntimeError(
                "gemm operand A should have >= 2 dimensions! Current shape: {}.".format(
                    a_shapes
                )
            )
        b_shapes = b._attrs["shape"]
        if len(b_shapes) != 2:
            raise RuntimeError(
                "gemm operand B should have 2 dimensions! Current shape: {}.".format(
                    b_shapes
                )
            )
        if not is_same_dtype(a.dtype(), b.dtype()):
            raise RuntimeError(
                "gemm operand A and B should have the same data type! Current A: {atype}, B: {btype}.".format(
                    atype=a.dtype(), btype=b.dtype()
                )
            )

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        """Call the gemm op.

        Parameters
        ----------
        a : Tensor
            Tensor with correct shape for the gemm operand A.
        b : Tensor
            Tensor with correct shape for the gemm operand B.

        Returns
        -------
        Tensor
            Output tensor for the gemm operation.
        """
        a, b = self._align_ab(a, b)
        self._attrs["inputs"] = [a, b]
        # TensorAccessor(b) is for bmm or rare cases of gemm where b is not constant weight
        self._attrs["input_accessors"] = [TensorAccessor(a), TensorAccessor(b)]
        self._set_depth()
        self._sanity_check(a, b)
        output_shape = self._infer_shapes(a, b)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output


def _profiler_results_groupby_key(instance):
    return (
        instance[1]["name"],  # unique op name
        instance[2],  # profiler executable
        instance[3],  # profiler key (gemm shape)
    )


def _profiler_group_reduce_min_key(group):
    return group[0][1]  # elapsed runtime


class GemmProfilerPostprocessingDelegate:
    """
    Object which collects profiler results after profiler executables complete,
    updates profiler results cache and the gemm nodes' attrs after all profilers complete.
    """

    def __init__(self):
        """
        Initialize storage for profiler results
        Instance=(
            ProfileResult=(best_algo, elapsed_runtime, workspace),
            func_attrs,
            profiler_filename,
            exec_key,
            split_k,
        )
        """
        self._instances = []

    def add_instance(self, instance: ProfileResult):
        """
        As a profiler executable completes, collect the result
        """
        self._instances.append(instance)

    def postprocess_results(self):
        """
        When all profiler executables complete, find the best instance
        (min runtime per op name, profiler executable and exec_key (i.e. gemm shape mnk)
        across multiple split_k values)
        The best instance is cached, and written into corresponding gemm nodes in the graph
        """
        target = backend.target.Target.current()
        for _, group in itertools.groupby(
            self._instances,
            key=_profiler_results_groupby_key,
        ):
            min_runtime_results = min(group, key=_profiler_group_reduce_min_key)
            (
                (best_algo, runtime, workspace),
                func_attrs,
                profiler_filename,
                exec_key,
                split_k,
            ) = min_runtime_results
            func_attrs["exec_path"][exec_key].algo = best_algo
            func_attrs["workspace"] = max(func_attrs["workspace"], workspace)
            func_attrs["split_k"] = split_k

            _LOGGER.info(
                f"Profiler ({profiler_filename} {exec_key}) selected kernel: "
                f"{best_algo=} {workspace=} {split_k=}",
            )

            tmp_op = next(iter(func_attrs["op_instance"].values()))
            exec_entry_sha1 = sha1(exec_key.encode("utf-8")).hexdigest()
            cache_record = GemmRecordEntry(
                exec_entry=exec_key,
                exec_entry_sha1=exec_entry_sha1,
                dtype_a=tmp_op.A.element.value,
                dtype_b=tmp_op.B.element.value,
                dtype_c=tmp_op.C.element.value,
                dtype_acc=tmp_op.accumulator_type().value,
                major_a=tmp_op.A.layout.value,
                major_b=tmp_op.B.layout.value,
                major_c=tmp_op.C.layout.value,
                op_type=func_attrs["op"],
                epilogue=tmp_op.epilogue_functor.value,
                device=target._arch,
                algo=best_algo,
                workspace=workspace,
                split_k=split_k,
                pshape=func_attrs["permute_shape"],
            )
            try:
                target.insert_profile_cache("gemm", cache_record.__dict__)
            except Exception as e:
                _LOGGER.warning(e)
