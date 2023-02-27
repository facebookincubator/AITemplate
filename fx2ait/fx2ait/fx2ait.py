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
import io
import logging
import os
import tempfile
import warnings
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import fx2ait.cache as cache

import torch

# @manual=//aitemplate/AITemplate/python/aitemplate:aitemplate
from aitemplate.testing import detect_target
from .converters.ait_converters import *  # isort:skip # noqa: F401 F403
from .converters.aten2ait_converters import *  # isort:skip # noqa: F401 F403
from aitemplate.compiler import compile_model
from aitemplate.compiler.base import _TorchConstantTensorData
from aitemplate.compiler.public import DynamicProfileStrategy, Tensor as AITTensor
from aitemplate.utils.serialization.serdes_code import dump_program, get_program
from torch.fx.node import _get_qualified_name
from torch.fx.passes.split_utils import getattr_recursive

from .converters.converter_registry import AIT_CONVERTERS
from .tensor_spec import TensorSpec

from .utils import dtype_to_str, make_str_ait_friendly

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AITInterpreterResult(NamedTuple):
    engine: Any
    input_names: Sequence[str]
    output_names: Sequence[str]
    fx_input_names: Sequence[str] = []


class AITInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        input_specs: List[TensorSpec],
        workdir: str,
        name: str,
        dll_name: str = "test.so",
        dynamic_profile_strategy=DynamicProfileStrategy.MAX,
        profile_devs=None,
        use_fp16_acc=True,
        dump_ait_dir: Optional[str] = None,
        keep_constants: Optional[bool] = None,
        load_ait_dir: Optional[str] = None,
        remote_cache_file_path: Optional[str] = None,
        save_remote_cache: Optional[bool] = False,
    ):
        """
        Args:
            module: target module for AITemplate compilation
            input_specs: sample input for the target module
            workdir: directory path for store AITemplate generated files
            name: directory name for store AITemplate generated files
            dll_name: AITemplate library name
            dynamic_profile_strategy: A dynamic profiling strategy, used to filter
            generated profiles at compile time.
            See also: :func:`~aitemplate.compiler.transform.profile.profile`
            use_fp16_acc: whether to uses fp16 accumulation for gemm ops.
            dump_ait_dir: AIT generated file dump location.
            keep_constants: whether to keep original constants or use random generated constants
            load_ait_dir: location for existing ait files
            remote_cache_file_path: AITemplate profiling cache location
            save_remote_cache: whether to save the updated cache
        """
        super().__init__(module)

        missing_ops = self.validate_conversion()
        if missing_ops:
            warnings.warn(
                "Interpretation will fail due to missing operations \n"
                + "\n".join(f"{i}" for i in missing_ops)
            )

        self.remote_cache_file_path = remote_cache_file_path
        self.save_remote_cache: bool = (
            True if save_remote_cache and self.remote_cache_file_path else False
        )
        self.remote_cache_bytes = self._load_profile_cache()
        if self.save_remote_cache:
            self.cache_dir = os.path.join(
                tempfile.mkdtemp(prefix="aitemplate_"), ".aitemplate"
            )
            os.environ["CACHE_DIR"] = self.cache_dir
            _LOGGER.info(f"Set CACHE_DIR to {self.cache_dir}")
        self.use_fp16_acc = use_fp16_acc
        self.hardware_target = self._create_target()
        self.input_specs = input_specs
        self.input_specs_iter = 0
        self.workdir = workdir
        self.name = name
        self.dll_name = dll_name
        self.dynamic_profile_strategy = dynamic_profile_strategy
        self.profile_devs = profile_devs

        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._fx_input_names: List[str] = []
        self._loaded_params: Dict[str, AITTensor] = {}

        self.dump_ait_dir = dump_ait_dir
        self.keep_constants = keep_constants
        self.load_ait_dir = load_ait_dir

    def _create_target(self):
        """Detect GPU target"""
        return detect_target(
            use_fp16_acc=self.use_fp16_acc, remote_cache_bytes=self.remote_cache_bytes
        )

    def _load_profile_cache(self) -> bytes:
        """
        Load AITemplate profile cache if cache file path is provided
        """
        if not self.remote_cache_file_path:
            return

        cache_bytes = io.BytesIO()
        cache.load_profile_cache(self.remote_cache_file_path, cache_bytes)
        remote_cache_bytes = cache_bytes.getvalue()
        _LOGGER.info(
            f"Loaded profile cache from remote: {self.remote_cache_file_path} with length {len(remote_cache_bytes)}",
        )
        return remote_cache_bytes

    def _upload_profile_cache(self, hardware_target) -> None:
        """
        Update AITemplate profile cache if cache file path is provided
        """
        cache_path = os.path.join(
            self.cache_dir, hardware_target._get_cache_file_name()
        )
        if not self.save_remote_cache or not cache_path:
            return

        _LOGGER.info(
            f"Uploading profile cache to remote: {self.remote_cache_file_path}",
        )
        cache.save_profile_cache(self.remote_cache_file_path, cache_path)
        _LOGGER.info(
            f"Upload AIT cache file to path {self.remote_cache_file_path} completed."
        )

    def validate_conversion(self):
        """
        Validate all node in target module has correspondent AIT converter support.
        """
        missing_converter = set()

        for node in self.module.graph.nodes:
            if node.op == "call_function" and not AIT_CONVERTERS.get(node.target):
                missing_converter.add(f"{node.op} {_get_qualified_name(node.target)}")
            elif node.op == "call_method" and not AIT_CONVERTERS.get(node.target):
                missing_converter.add(f"{node.op} torch.Tensor.{node.target}")
            elif node.op == "call_module":
                submod = self.fetch_attr(node.target)
                submod_type = getattr(submod, "_base_class_origin", type(submod))
                if not AIT_CONVERTERS.get(submod_type):
                    missing_converter.add(f"{node.op} {torch.typename(submod_type)}")

        return missing_converter

    def run(self) -> AITInterpreterResult:
        """
        Build AITemplate engine
        Returns:
        Compiled AITemplate engine packaged as AITInterpreterResult
        """
        run_module_start_time = datetime.now()
        output_tensors = super().run()
        _LOGGER.info(
            f"Run Module elapsed time: {datetime.now() - run_module_start_time}"
        )
        # FX2AIT name if composed as MODULE_NAME/submodule_name, we put all profile file on
        # parent dir of submodule_name to share across submodules.
        profile_dir = (
            os.path.join(self.workdir, self.name[0 : self.name.rindex("/")])
            if self.name.find("/") != -1
            else self.workdir
        )
        args = {
            "tensor": output_tensors,
            "target": self.hardware_target,
            "workdir": self.workdir,
            "test_name": self.name,
            "profile_devs": self.profile_devs,
            "dynamic_profiling_strategy": self.dynamic_profile_strategy,
            "dll_name": self.dll_name,
            "profile_dir": profile_dir,
        }
        if self.dump_ait_dir:
            dump_ait_path = os.path.join(self.dump_ait_dir, self.name + ".py")
            random_constants = not self.keep_constants
            dump_program(
                output_tensors, dump_ait_path, random_constants=random_constants
            )
            _LOGGER.info(f"Dumped AIT model to {dump_ait_path}")

        if self.load_ait_dir:
            load_ait_path = os.path.join(self.load_ait_dir, self.name + ".py")
            _LOGGER.info(f"Loaded AIT model from {load_ait_path}")
            output_tensors, _ = get_program(load_ait_path)
            if isinstance(output_tensors, AITTensor):
                output_tensors = (output_tensors,)
            args["tensor"] = output_tensors

        self.engine = compile_model(**args)
        ait_input_names = [
            n._attrs["name"]
            for n in self.engine.debug_sorted_graph
            if n._attrs["is_input"]
        ]
        for name in ait_input_names:
            assert (
                self._fx_input_names.count(name) == 1
            ), f"Cannot find AIT's compiled input: {name} in fx graph!"

        for name in self._fx_input_names:
            if name in ait_input_names:
                self._input_names.append(name)

        for i, input_name in enumerate(self._fx_input_names):
            _LOGGER.info("Set input{}: {}".format(i, input_name))

        if self.engine is None:
            raise RuntimeError("Engine is missing!")

        if self.save_remote_cache:
            self._upload_profile_cache(self.hardware_target)

        return AITInterpreterResult(
            self.engine,
            self._input_names,
            self._output_names,
            self._fx_input_names,
        )

    def run_node(self, n):
        self._cur_node_name = str(n)
        return super().run_node(n)

    def placeholder(self, target, args, kwargs):
        self._fx_input_names.append(target)
        input_spec = self.input_specs[self.input_specs_iter]
        self.input_specs_iter += 1

        return AITTensor(
            shape=input_spec.shape,
            dtype=dtype_to_str(input_spec.dtype),
            name=target,
            is_input=True,
        )

    def get_attr(self, target, args, kwargs):
        attr_val = getattr_recursive(self.module, target)

        if not isinstance(attr_val, (torch.Tensor, torch.nn.Parameter)):
            raise RuntimeError(f"Unexpected get_attr value for {target}: {attr_val}")

        ait_friendly_name = make_str_ait_friendly(target)
        ait_dtype = dtype_to_str(attr_val.dtype)
        ait_val = attr_val.contiguous()
        if ait_friendly_name in self._loaded_params:
            existing_tensor = self._loaded_params[ait_friendly_name]
            assert existing_tensor._attrs["dtype"] == ait_dtype
            assert existing_tensor._attrs["data"].tensor == ait_val
            return existing_tensor

        data = _TorchConstantTensorData(ait_val)
        tensor = AITTensor(
            shape=attr_val.shape, dtype=ait_dtype, name=ait_friendly_name
        )
        tensor._bind_data(data)
        self._loaded_params[ait_friendly_name] = tensor
        return tensor

    def call_function(self, target, args, kwargs):
        converter = AIT_CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(
                f"Conversion of function {torch.typename(target)} not currently supported!"
            )

        assert self._cur_node_name is not None
        return converter(target, args, kwargs, self._cur_node_name)

    def call_method(self, target, args, kwargs):
        assert isinstance(target, str)
        converter = AIT_CONVERTERS.get(target)

        if not converter:
            raise RuntimeError(
                f"Conversion of method {target} not currently supported!"
            )

        assert self._cur_node_name is not None
        return converter(target, args, kwargs, self._cur_node_name)

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        converter = AIT_CONVERTERS.get(submod_type)

        if not converter:
            raise RuntimeError(
                f"Conversion of module of type {submod_type} not currently supported!"
            )

        assert self._cur_node_name is not None
        return converter(target, submod, args, kwargs, self._cur_node_name)

    def output(self, target, args, kwargs):
        assert len(args) == 1
        if isinstance(args[0], tuple):
            outputs = args[0]
        elif isinstance(args[0], list):
            outputs = tuple(args[0])
        else:
            outputs = (args[0],)

        for i, output in enumerate(outputs):
            name = f"output_{i}"
            output._attrs["name"] = name
            output._attrs["is_output"] = True
            self._output_names.append(name)

        return outputs
