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
Util functions for CUDA codegen.
"""
import logging

from aitemplate.backend import registry

from aitemplate.utils.mk_cutlass_lib.mk_cutlass_lib import mk_cutlass_lib

# pylint: disable=C0103,C0415,W0707


_LOGGER = logging.getLogger(__name__)


class Args:
    def __init__(self, arch):
        self.operations = "all"
        self.build_dir = ""
        self.curr_build_dir = ""
        self.generator_target = ""
        self.architectures = arch
        self.kernels = "all"
        self.ignore_kernels = ""
        self.cuda_version = "11.4.0"
        self.kernel_filter_file = None
        self.selected_kernel_list = None
        self.interface_dir = None
        self.filter_by_cc = True
        self.disable_full_archs_compilation = False


registry.reg("cuda.make_cutlass_lib")(mk_cutlass_lib)


@registry.reg("cuda.gen_cutlass_ops")
def gen_ops(arch):
    import cutlass_lib

    args = Args(arch)
    manifest = cutlass_lib.manifest.Manifest(args)
    try:
        func = getattr(cutlass_lib.generator, "GenerateSM" + arch)
        func(manifest, args.cuda_version)
    except AttributeError as e:
        raise NotImplementedError(
            "Arch " + arch + " is not supported by current cutlass lib."
        ) from e
    try:
        func = getattr(cutlass_lib.extra_operation, "GenerateSM" + arch)
        func(manifest, args)
    except AttributeError:
        _LOGGER.warning("Arch " + arch + " is not supported by extra ops.")
    return manifest.operations
