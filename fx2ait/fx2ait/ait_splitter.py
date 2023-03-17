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
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.fx.passes.operator_support as ops
import torch.fx.passes.splitter_base as splitter_base
from fx2ait.acc_tracer import acc_ops
from fx2ait.ait_module import AITModule

from fx2ait.converters.converter_registry import AIT_CONVERTERS
from fx2ait.fx2ait import AITInterpreter
from torch.fx.passes.operator_support import create_op_support, OperatorSupportBase
from torch.fx.passes.tools_common import get_acc_ops_name

try:
    torch.ops.load_library("//deeplearning/ait:AITModel")
except BaseException:
    torch.ops.load_library("build/libait_model.so")


_VIEW_OPS = frozenset(
    (
        acc_ops.unsqueeze,
        acc_ops.squeeze,
        acc_ops.reshape,
        acc_ops.flatten,
    )
)

DEFAULT_MIN_ACC_MODULE_SIZE = 10


def _decline_if_would_trigger_extra_copies(
    has_converter: OperatorSupportBase,
) -> OperatorSupportBase:
    def _impl(
        submodules: Mapping[str, torch.nn.Module],
        node: torch.fx.Node,
    ):
        def _any_supported(nodes: Sequence[torch.fx.Node]) -> bool:
            return any(
                has_converter.is_node_supported(submodules, node) for node in nodes
            )

        if node.target not in _VIEW_OPS:
            return True

        if _any_supported(node.users) or _any_supported(node.all_input_nodes):
            return True

        return False

    return create_op_support(_impl)


def create_ait_operator_support(
    use_implicit_batch_dim=True,
    op_lowering_disallow_list=None,
    allow_int_inputs=False,
    allow_op_supports=None,
) -> ops.OperatorSupportBase:
    """Creates an `OperatorSupportBase` instance used for AIT splitting purpose."""
    # Create an `OperatorSupport` that declares a node supported if it
    # finds a registered AIT converter.
    support_dict: Dict[str, None] = {}
    for k in AIT_CONVERTERS.keys():
        # may need to switch the op name here
        support_dict[get_acc_ops_name(k)] = None
    supported_if_converter_registered = ops.OperatorSupport(support_dict=support_dict)

    op_lowering_disallow_set = (
        set() if op_lowering_disallow_list is None else set(op_lowering_disallow_list)
    )
    chained_not_supported_ops = (
        []
        if allow_int_inputs
        else [
            ops.OpSupports.decline_if_input_dtype(torch.int64),
            ops.OpSupports.decline_if_input_dtype(torch.int32),
        ]
    )
    chained_not_supported_ops += [
        ops.OpSupports.decline_if_node_in_names(op_lowering_disallow_set),
        # 1. We only support subgraphs with torch.Tensor inputs for now
        ops.OpSupports.decline_if_input_dtype(torch.float64),
        ops.OpSupports.decline_if_input_dtype(dict),
        # 2. Node is supported if it has AIT converter:
        supported_if_converter_registered,
        # 3. Decline nodes that would trigger extra copies. This can happen if
        # we have an output that is just a view of an input, for example.
        # Note that this is not required for correctness, it is merely an
        # optimization.
        _decline_if_would_trigger_extra_copies(supported_if_converter_registered),
    ]
    if allow_op_supports:
        return ops.any_chain(ops.chain(*chained_not_supported_ops), *allow_op_supports)
    return ops.chain(*chained_not_supported_ops)


class AITSplitterSettings(splitter_base._SplitterSettingBase):
    # TODO: Fix this once pytorch nightly is updated
    def __init__(
        self, min_acc_module_size=DEFAULT_MIN_ACC_MODULE_SIZE, allow_int_inputs=False
    ):
        super().__init__()
        self.min_acc_module_size = min_acc_module_size
        self.exclude_support_node_name: set = set()
        self.allow_int_inputs: bool = allow_int_inputs


class AITSplitter(splitter_base._SplitterBase):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Sequence[Any],
        operator_support: ops.OperatorSupportBase = None,
        settings: AITSplitterSettings = None,
    ):
        if not settings:
            settings = AITSplitterSettings()
        if not operator_support:
            operator_support = create_ait_operator_support(
                op_lowering_disallow_list=settings.exclude_support_node_name,
                allow_int_inputs=settings.allow_int_inputs,
            )
        else:
            operator_support = ops.chain(
                operator_support,
                ops.OpSupports.decline_if_node_in_names(
                    settings.exclude_support_node_name
                ),
            )
        super().__init__(
            module,
            sample_input,
            operator_support,
            settings,
            non_acc_submodule_name="_run_on_gpu_",
        )

    def _lower_model_to_backend(
        self, mod: torch.fx.GraphModule, inputs: Iterable[torch.Tensor]
    ):
        """
        Lower a GraphModule `mod` to AITemplate with `inputs`.
        """
        # Current code for lowering is place-holder, subject to future change
        # based on feeds model's actual status
        interp = AITInterpreter(mod, [inputs])
        interpreter_result = interp.run(*inputs)
        return AITModule(
            torch.classes.fb.AITModel(
                interpreter_result.engine.lib_path,
                interpreter_result.input_names,
                interpreter_result.output_names,
                torch.float16,
                torch.float,
                1,  # num_runtimes
            ),
            interpreter_result,
        )

    # TODO add _find_culprit once minimizer completed
