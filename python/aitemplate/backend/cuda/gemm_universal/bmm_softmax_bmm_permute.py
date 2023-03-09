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

from aitemplate.backend import registry


@registry.reg("cuda.bmm_softmax_bmm_permute.func_decl")
def gen_function_decl(func_attrs):
    raise NotImplementedError("bmm_softmax_bmm_permute kernel is not implemented.")


@registry.reg("cuda.bmm_softmax_bmm_permute.gen_function")
def gen_function(func_attrs):
    raise NotImplementedError("bmm_softmax_bmm_permute kernel is not implemented.")


@registry.reg("cuda.bmm_softmax_bmm_permute.func_call")
def gen_function_call(func_attrs, indent="  "):
    raise NotImplementedError("bmm_softmax_bmm_permute kernel is not implemented.")
