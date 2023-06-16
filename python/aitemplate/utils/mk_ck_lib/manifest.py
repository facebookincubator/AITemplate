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

#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import os.path
import re

from aitemplate.utils.mk_ck_lib.library import OperationKind, OperationKindNames


class Manifest:
    def __init__(self, args=None):
        self.operations = {}
        self.args = args
        self.operation_count = 0
        self.operations_by_name = {}

        self.kernel_filter = ""
        self.kernel_filter_list = []
        self.kernel_names = []
        self.operations_enabled = []
        self.selected_kernels = []
        self.ignore_kernels = []
        # self.compute_capabilities = [50,]
        self.curr_build_dir = "."
        # self.filter_by_cc = True

        if self.args:
            self.kernel_filter = self.args.kernels
            self.curr_build_dir = args.curr_build_dir

            if args.operations == "all":
                self.operations_enabled = []
            else:
                operations_list = [
                    OperationKind.Gemm,
                    OperationKind.Conv2d,
                    OperationKind.Softmax,
                ]
                self.operations_enabled = [
                    x
                    for x in operations_list
                    if OperationKindNames[x] in args.operations.split(",")
                ]

            if args.kernels == "all":
                self.kernel_names = []
            else:
                self.kernel_names = [x for x in args.kernels.split(",") if x != ""]

            self.ignore_kernels = [x for x in args.ignore_kernels.split(",") if x != ""]

            if args.kernel_filter_file is None:
                self.kernel_filter_list = []
            else:
                self.kernel_filter_list = self.get_kernel_filters(
                    args.kernel_filter_file
                )

    def get_kernel_filters(self, kernelListFile):
        if os.path.isfile(kernelListFile):
            with open(kernelListFile, "r") as fileReader:
                lines = [
                    line.rstrip() for line in fileReader if not line.startswith("#")
                ]

            lines = [re.compile(line) for line in lines if line]
            return lines
        else:
            return []

    def filter_out_kernels(self, kernel_name, kernel_filter_list):
        for kernel_filter_re in kernel_filter_list:
            if kernel_filter_re.search(kernel_name) is not None:
                return True

        return False

    def _filter_string_matches(self, filter_string, haystack):
        """Returns true if all substrings appear in the haystack in order"""
        substrings = filter_string.split("*")
        for sub in substrings:
            idx = haystack.find(sub)
            if idx < 0:
                return False
            haystack = haystack[idx + len(sub) :]
        return True

    def filter(self, operation):
        """Filtering operations based on various criteria"""
        enabled = True

        if (
            len(self.operations_enabled)
            and operation.operation_kind not in self.operations_enabled
        ):
            return False
        # eliminate duplicates
        if str(operation) in self.operations_by_name.keys():
            return False
        # Filter based on list of valid substrings
        if len(self.kernel_names):
            name = str(operation)
            enabled = False

            # compare against the include list
            for name_substr in self.kernel_names:
                if self._filter_string_matches(name_substr, name):
                    enabled = True
                    break

            # compare against the exclude list
            for name_substr in self.ignore_kernels:
                if self._filter_string_matches(name_substr, name):
                    enabled = False
                    break

        if len(self.kernel_filter_list) > 0:
            enabled = False
            if self.filter_out_kernels(str(operation), self.kernel_filter_list):
                enabled = True

        # todo: filter based on compute data type
        return enabled

    def append(self, operation):
        """
        Inserts the operation.
        operation_kind -> configuration_name -> []
        """
        if self.filter(operation):
            self.selected_kernels.append(str(operation))

            self.operations_by_name[str(operation)] = operation

            # add the configuration
            configuration_name = str(operation)

            if operation.operation_kind not in self.operations.keys():
                self.operations[operation.operation_kind] = {}
            if (
                operation.extra_kind
                not in self.operations[operation.operation_kind].keys()
            ):
                self.operations[operation.operation_kind][operation.extra_kind] = {}

            if (
                configuration_name
                not in self.operations[operation.operation_kind][
                    operation.extra_kind
                ].keys()
            ):
                self.operations[operation.operation_kind][operation.extra_kind][
                    configuration_name
                ] = []

            self.operations[operation.operation_kind][operation.extra_kind][
                configuration_name
            ].append(operation)
            self.operation_count += 1
