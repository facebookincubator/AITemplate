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
Templates for different GeMM epilogues.
"""
from typing import Dict, List, NamedTuple

from aitemplate.compiler.ops.common.epilogue import EpilogueOp


class GeMMEpilogueSpec(NamedTuple):
    """
    Base class for GeMM epilogue templates.
    """

    dtype_to_config_filenames: Dict[str, List[str]]
    element_op_template: str
    arguments_template: str = ""
    dtype_to_instantiation_template: Dict[str, str] = {"float16": ""}
    dtype_to_profiler_call_template: Dict[str, str] = {"float16": ""}
    func_decl_template: str = ""
    func_call_template: str = ""


class GeMMBiasEpilogueSpec(GeMMEpilogueSpec):
    """
    Base class for Activation(GeMM + Bias) templates.
    """

    arguments_template: str = """
{{indent}}                                static_cast<BiasDataType*>(bias_ptr),
"""
    dtype_to_instantiation_template: Dict[str, str] = {
        "float16": """
DeviceMem b(sizeof(ck::half_t) * n);
""",
    }
    dtype_to_profiler_call_template: Dict[str, str] = {
        "float16": """
      (ck::half_t*) b.GetDeviceBuffer(),
""",
    }
    func_decl_template: str = """
  {{dtype}}* bias_ptr,
"""
    func_call_template: str = """
{{indent}}    {{bias_ptr}},
"""


_GEMM_EPILOGUE_TO_SPEC = {
    EpilogueOp.NA: GeMMEpilogueSpec(
        dtype_to_config_filenames={
            "float16": [
                "device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp"  # pylint: disable=C0301
            ],
        },
        element_op_template="""
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = PassThrough;
        """,
    ),
    EpilogueOp.BIAS: GeMMBiasEpilogueSpec(
        dtype_to_config_filenames={
            "float16": [
                "device_operation/src/device_gemm_xdl_c_shuffle_bias_add_f16_f16_f16_mk_nk_mn_instance.cpp"  # pylint: disable=C0301
            ],
        },
        element_op_template="""
using BiasAdd = ck::tensor_operation::element_wise::BiasAdd;
using OutElementOp = BiasAdd;
        """,
    ),
}


def get_epilogue_spec(epilogue_op: EpilogueOp) -> GeMMEpilogueSpec:
    return _GEMM_EPILOGUE_TO_SPEC[epilogue_op]
