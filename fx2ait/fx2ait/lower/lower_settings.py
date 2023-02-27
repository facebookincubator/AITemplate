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
from enum import Enum
from typing import Any, List, Optional, Set, Type

import torch

from aitemplate.compiler.public import DynamicProfileStrategy
from torch import nn


class LowerPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dc.dataclass
class LowerSettings:
    """
    Basic configuration for lowering stack.
    Args:
    max_batch_size: The maximium batch_size for the input
    min_acc_module_size: minimal number of nodes for individual accelerated submodule.
    workdir: the working directory path.
    name: the working directory name.
    dll_name: AITemplate generated .so file name
    dynamic_profile_strategy:
        A dynamic profiling strategy, used to filter
        generated profiles at compile time.
    precision: The runtime precision setting
    use_fp16_acc:
        For LowerPrecision.FP16, use_fp16_acc can be either True or False.
        use_fp16_acc=True uses fp16 accumulation for gemm ops.
        use_fp16_acc=False uses fp32 accumulation for gemm ops.
        Set use_fp16_acc=True for better perf; set use_fp16_acc=False for better accuracy.
        For LowerPrecision.FP32, use_fp16_acc is invalid.
    leaf_module_list: The list of modules that acc_tracer will not trace into.
    output_precision: The AITemplate output precision level.
    additional_inputs: The additional input to help determine input batch_size dimension range.
    remote_cache_file_path: Location for AITemplate cache file.
    save_remote_cache: Whether to save the current cache update to the cache file.
    dump_ait_dir: Dump AIT module into python code
    keep_constants: Whether or not to keep the constants in the dumped AIT module
    load_ait_dir: Reload AIT module from dumped AIT python code instead.
    """

    max_batch_size: int = 2048
    min_acc_module_size: int = 10
    workdir: str = ""
    name: str = ""
    dll_name: str = "ait_engine.so"
    dynamic_profile_strategy: DynamicProfileStrategy = DynamicProfileStrategy.MAX
    profile_devs: Any = None
    # If None, infer the dtypes from the sample inputs.
    precision: Optional[LowerPrecision] = LowerPrecision.FP16
    use_fp16_acc: bool = True  # only valid for precision == FP16
    ast_rewriter_allow_list: Optional[Set[Type[nn.Module]]] = None
    leaf_module_list: Optional[Set[Type[nn.Module]]] = None
    # If None, infer the dtypes from the sample inputs.
    output_precision: Optional[LowerPrecision] = LowerPrecision.FP16
    additional_inputs: Optional[List[torch.Tensor]] = None
    remote_cache_file_path: Optional[str] = None
    save_remote_cache: Optional[bool] = None
    dump_ait_dir: Optional[str] = None
    keep_constants: Optional[bool] = None
    load_ait_dir: Optional[str] = None
    # jit.trace AITModule
    trace_ait_module: bool = True
