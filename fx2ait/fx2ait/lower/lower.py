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
import dataclasses as dc
import datetime
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence

import fx2ait.acc_tracer.acc_tracer as acc_tracer

import torch

from fx2ait.acc_tracer.ait_acc_normalizer import update_acc_op_mappers_for_ait
from fx2ait.ait_module import AITModule

from fx2ait.ait_splitter import AITSplitter, AITSplitterSettings
from fx2ait.fx2ait import AITInterpreter, AITInterpreterResult
from fx2ait.tensor_spec import TensorSpec
from torch import fx, nn
from torch.fx.passes.splitter_base import generate_inputs_for_submodules, SplitResult

from .lower_settings import LowerPrecision, LowerSettings

logger: logging.Logger = logging.getLogger(__name__)
Input = Sequence[Any]

torch.ops.load_library("build/libait_model.so")


# A list of (function, target) pairs to not apply acc normalization
# to when scripting. For one reason or another, these targets do
# not play well with TorchScript after normalization.
SCRIPTING_ACC_NORMALIZATION_BLOCKLIST = {
    ("call_function", operator.getitem),
    ("call_method", "to"),
}


@dc.dataclass
class AitLowerInterpreter:
    lower_settings: LowerSettings

    @classmethod
    def create(cls, lower_settings):
        return AitLowerInterpreter(lower_settings)

    def __call__(
        self,
        module_name: str,
        mod: fx.GraphModule,
        inputs: List[torch.Tensor],
    ) -> AITInterpreterResult:
        (additional_inputs,) = self.lower_settings.additional_inputs
        if additional_inputs is None:
            input_specs = TensorSpec.from_input_list_with_batch_size(
                inputs, self.lower_settings.max_batch_size
            )
        else:
            input_specs = TensorSpec.from_two_input_lists(inputs, additional_inputs)
        logger.info("Input specs: %s", input_specs)

        interpreter = AITInterpreter(
            module=mod,
            input_specs=input_specs,
            workdir=self.lower_settings.workdir,
            name=f"{self.lower_settings.name}/{module_name}",
            dll_name=module_name + "-" + self.lower_settings.dll_name,
            dynamic_profile_strategy=self.lower_settings.dynamic_profile_strategy,
            profile_devs=self.lower_settings.profile_devs,
            use_fp16_acc=self.lower_settings.use_fp16_acc,
            remote_cache_file_path=self.lower_settings.remote_cache_file_path,
            save_remote_cache=self.lower_settings.save_remote_cache,
            dump_ait_dir=self.lower_settings.dump_ait_dir,
            keep_constants=self.lower_settings.keep_constants,
            load_ait_dir=self.lower_settings.load_ait_dir,
        )

        interp_result: AITInterpreterResult = interpreter.run()

        return interp_result


def create_ait_lower_interpreter(lower_settings: LowerSettings) -> AitLowerInterpreter:
    return AitLowerInterpreter.create(lower_settings)


def default_split_function(
    model: fx.GraphModule, inputs: Input, lower_settings: LowerSettings
) -> SplitResult:
    settings = AITSplitterSettings(
        min_acc_module_size=lower_settings.min_acc_module_size,
        allow_int_inputs=lower_settings.allow_int_inputs,
    )
    splitter = AITSplitter(model, inputs, settings=settings)
    splitter.node_support_preview()
    return splitter.generate_split_results()


def default_lower_pass(
    create_ait_interpreter: Callable[[LowerSettings], AitLowerInterpreter],
) -> Callable:
    def lower_pass(
        mod: nn.Module, input: Input, lower_settings: LowerSettings, module_name: str
    ) -> nn.Module:
        """
        Create a module transformation pass which lowers an `nn.Module` into an
        `AITModule`
        """
        interpreter = create_ait_interpreter(lower_settings)
        interp_res: AITInterpreterResult = interpreter(module_name, mod, input)

        # Return a scriptable module since some use cases need to script the top
        # level module
        return AITModule.create_ait_module_wrapper(
            torch.classes.ait.AITModel(
                interp_res.engine.lib_path,
                interp_res.input_names,
                interp_res.output_names,
                _precision_to_torch_type(lower_settings.precision),
                _precision_to_torch_type(lower_settings.output_precision),
                1,  # num_runtimes
            ),
            interp_res,
            lower_settings.trace_ait_module,
            *input,
        )

    return lower_pass


@dc.dataclass(frozen=True)
class AitLowerer:
    """Lowers a module using fx2ait.

    This is a composable class to facilitate fx2ait. A normal fx2ait process
    composes of the following passes to transform an `fx.GraphModule`:

        1. trace - use torch.fx to trace the module so we can get the graph
            representation of the model.
        2. split - the graph module is split into several submodules,
            running either via AITemplate, or via regular CUDA.

    For each split that need to run via AIT, the following passes are
    invoked:

        3. `AITInterpreter` - build the AIT engine for the submodule that
            can be supported through `AITInterpreter`.
        4. Wraps the executable AIT engine into `AITModule`, which is an `nn.Module`.
        5. The converted submodule is then set back onto the top-level module

    """

    lower_settings: LowerSettings
    lower_pass: Callable
    static_deps_initialized: bool = False

    @staticmethod
    def initialize_static_deps() -> None:
        if AitLowerer.static_deps_initialized:
            logger.info("Static deps were initialized already")
        else:
            logger.info("Initializing static deps")
            update_acc_op_mappers_for_ait()
            AitLowerer.static_deps_initialized = True
            logger.info("Initialized static deps")

    @classmethod
    def create(
        cls,
        lower_settings: LowerSettings,
        interpreter_builder: Callable = create_ait_lower_interpreter,
    ) -> "AitLowerer":
        """Instantiate an `AitLowerer` instance."""
        cls.initialize_static_deps()

        return cls(
            lower_settings=lower_settings,
            lower_pass=default_lower_pass(create_ait_lower_interpreter),
        )

    def lower_func(
        self, split_result: SplitResult, additional_inputs: Optional[Input] = None
    ) -> nn.Module:
        if additional_inputs:
            additional_submodule_inputs = generate_inputs_for_submodules(
                split_result.split_module,
                additional_inputs,
                list(split_result.submodule_inputs.keys()),
            )
        else:
            additional_submodule_inputs = None

        for submod_name, submod_inputs in split_result.submodule_inputs.items():
            submod = getattr(split_result.split_module, submod_name)
            # Only acc submodules will be lowered.
            if not submod_name.startswith(split_result.non_acc_submodule_prefix):
                logger.info(f"Now lowering submodule {submod_name}")
                lowering_start_time = datetime.datetime.now()

                self.lower_settings.additional_inputs = (
                    additional_submodule_inputs[submod_name]
                    if additional_submodule_inputs
                    else None,
                )

                lowered_module = self.lower_pass(
                    submod, submod_inputs, self.lower_settings, submod_name
                )
                setattr(split_result.split_module, submod_name, lowered_module)
                logger.info(
                    f"Lowering submodule {submod_name} elapsed time {datetime.datetime.now() - lowering_start_time}"
                )

        return split_result.split_module

    def __call__(
        self,
        module: nn.Module,
        inputs: Input,
        additional_inputs: Optional[Input] = None,
    ) -> nn.Module:
        module.eval()
        module = acc_tracer.trace(
            module, inputs, leaf_module_list=self.lower_settings.leaf_module_list
        )
        split_result = default_split_function(module, inputs, self.lower_settings)
        lower_result = self.lower_func(split_result, additional_inputs)

        return lower_result


def _precision_to_torch_type(
    precision: Optional[LowerPrecision],
) -> Optional[torch.dtype]:
    if precision == LowerPrecision.FP16:
        return torch.float16
    elif precision == LowerPrecision.FP32:
        return torch.float
    elif precision == LowerPrecision.INT8:
        return torch.int8
    return None
