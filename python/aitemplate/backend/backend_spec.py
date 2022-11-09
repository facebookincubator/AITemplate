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
Backend Specifications.
"""

from dataclasses import dataclass, field

from typing import Dict, List, Tuple

import jinja2

from ..compiler.ops.common.epilogue import FuncEnum
from .target import Target


class BackendSpec:
    pass


@dataclass
class CPUBackendSpec(BackendSpec):
    func_enum_to_func_name: Dict[FuncEnum, str] = field(
        default_factory=lambda: {
            FuncEnum.ADD: "+",
            FuncEnum.SUB: "-",
            FuncEnum.MUL: "*",
            FuncEnum.DIV: "/",
        }
    )


@dataclass
class GPUBackendSpec(BackendSpec):
    dtype_to_backend_fp16_dtype: Dict[str, str] = field(
        default_factory=lambda: {
            "float16": "half",
        }
    )

    dtype_to_backend_dtype: Dict[str, str] = field(
        default_factory=lambda: {
            "float16": "half",
            "float": "float",
            "int64": "int64_t",
        }
    )

    backend_datatype_convertors: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "half": {"float": "__half2float"},
            "float": {"half": "__float2half_rn"},
        }
    )

    read_num_elements_to_backend_type: List[Tuple[int, str]] = field(
        default_factory=lambda: [
            (8, "uint4"),
            (4, "uint2"),
            (2, "uint"),
            (1, "half"),
        ]
    )
    op_num_elements_to_backend_type: List[Tuple[int, str]] = field(
        default_factory=lambda: [
            (2, "half2"),
            (1, "half"),
        ]
    )
    op_type_priority_list: List[str] = field(
        default_factory=lambda: [
            "half2",
            "half",
            "float",
        ]
    )
    func_enum_to_func_name: Dict[FuncEnum, Dict[str, str]] = field(
        default_factory=lambda: {
            FuncEnum.ADD: {
                "half2": "__hadd2",
                "half": "__hadd",
                "float": "__fadd_rn",
            },
            FuncEnum.SUB: {
                "half2": "__hsub2",
                "half": "__hsub",
                "float": "__fsub_rn",
            },
            FuncEnum.MUL: {
                "half2": "__hmul2",
                "half": "__hmul",
                "float": "__fmul_rn",
            },
            FuncEnum.DIV: {
                "half2": "__h2div",
                "half": "__hdiv",
                "float": "__fdiv_rn",
            },
            FuncEnum.COS: {
                "half2": "h2cos",
                "half": "hcos",
                "float": "cosf",
            },
            FuncEnum.SIN: {
                "half2": "h2sin",
                "half": "hsin" if Target.current().name() == "cuda" else "hsin_custom",
                "float": "sinf",
            },
            FuncEnum.TANH: {
                "half2": "fast_tanh",
                "half": "fast_tanh",
                "float": "tanh",
            },
            FuncEnum.ABS: {
                "half2": "__habs2",
                "half": "__habs",
                "float": "fabsf",
            },
            FuncEnum.LOGE: {
                "half2": "h2log",
                "half": "hlog",
                "float": "logf",
            },
            FuncEnum.EXP: {
                "half2": "h2exp",
                "half": "hexp",
                "float": "expf",
            },
            FuncEnum.SQRT: {
                "half2": "h2sqrt",
                "half": "hsqrt",
                "float": "sqrtf",
            },
            FuncEnum.MAX: {
                "half2": "hmax2_nan",
                "half": "hmax_nan",
                "float": "fmaxf_nan",
            },
            FuncEnum.MIN: {
                "half2": "hmin2_nan",
                "half": "hmin_nan",
                "float": "fminf_nan",
            },
            FuncEnum.SIGN: {
                "half2": "h2sign_custom",
                "half": "sign_custom<half>",
                "float": "sign_custom<float>",
            },
            FuncEnum.SIGMOID: {
                "half2": "h2sigmoid_custom",
                "half": "hsigmoid_custom",
                "float": "fsigmoid_custom",
            },
            FuncEnum.LRELU: {
                "half2": "leaky_relu",
                "half": "leaky_relu",
                "float": "leaky_relu",
            },
            FuncEnum.HARDTANH: {
                "half2": "h2hard_tanh",
                "half": "hard_tanh<half>",
                "float": "hard_tanh<float>",
            },
            FuncEnum.RELU: {"half2": "relu", "half": "relu", "float": "relu"},
            FuncEnum.NAN_TO_NUM: {
                "half2": "nan_to_num",
                "half": "nan_to_num",
                "float": "nan_to_num",
            },
            FuncEnum.CLAMP_NAN_TO_NUM: {
                "half2": "clamp_nan_to_num",
                "half": "clamp_nan_to_num",
                "float": "clamp_nan_to_num",
            },
            FuncEnum.SILU: {
                "half2": "h2silu",
                "half": "hsilu",
                "float": "fsilu",
            },
            FuncEnum.POW: {
                "half2": "h2pow",
                "half": "hpow",
                "float": "fpow",
            },
            FuncEnum.GELU: {
                "half": "hgelu",
                "float": "fgelu",
            },
            FuncEnum.FASTGELU: {
                "half": "h_fast_gelu",
                "float": "f_fast_gelu",
            },
            FuncEnum.SOFTPLUS: {
                "half2": "h2softplus",
                "half": "hsoftplus",
                "float": "fsoftplus",
            },
        }
    )

    def get_backend_type(
        self,
        num_elements: int,
        dtype: str,
        num_elements_to_backend_type_list: List[Tuple[int, str]],
    ) -> str:
        if dtype not in ("float16", "float"):
            raise NotImplementedError("Unsupported dtype {}!".format(dtype))
        for alignment, backend_type in num_elements_to_backend_type_list:
            if num_elements % alignment == 0:
                return backend_type
        raise RuntimeError(
            "Failed to infer data type! num_elements: {}, num_elements_to_backend_type_list: {}".format(
                num_elements, num_elements_to_backend_type_list
            )
        )

    def get_candidate_op_types(self, op_t: str) -> List[str]:
        res = []
        found = False
        for t in self.op_type_priority_list:
            if t == op_t:
                found = True
            if found:
                res.append(t)
        return res

    def get_dtype_to_dtype(self, dtype: str, type_dict: Dict[str, str]):
        data_type = type_dict.get(dtype)
        if not data_type:
            raise NotImplementedError("Unsupported dtype {}!".format(dtype))
        return data_type

    def get_fp16_dtype(self, dtype: str):
        return self.get_dtype_to_dtype(dtype, self.dtype_to_backend_fp16_dtype)

    def dtype_to_backend_type(self, dtype: str):
        return self.get_dtype_to_dtype(dtype, self.dtype_to_backend_dtype)

    def dtype_to_lib_type(self, dtype: str):
        raise NotImplementedError


@dataclass
class ROCMSpec(GPUBackendSpec):
    backend_name = "rocm"
    index_type = "int64_t"
    prefix = "hip"
    stream = "stream"
    cub = "hipcub"

    cast_to_half_ptr_template = jinja2.Template("reinterpret_cast<half*>({{name}})")
    cast_to_const_half_ptr_template = jinja2.Template(
        "reinterpret_cast<const half*>({{name}})"
    )
    header_src_template = jinja2.Template(
        """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
{{extra_header}}
        """
    )
    half2_data_ref = ".data"

    dtype_to_ck_type: Dict[str, str] = field(
        default_factory=lambda: {
            "float16": "ck::half_t",
            "float": "float",
        }
    )

    def dtype_to_lib_type(self, dtype: str):
        return self.get_dtype_to_dtype(dtype, self.dtype_to_ck_type)


@dataclass
class CUDASpec(GPUBackendSpec):
    backend_name = "cuda"
    index_type = "int64_t"
    prefix = "cuda"
    stream = "stream"
    cub = "cub"

    cast_to_half_ptr_template = jinja2.Template("reinterpret_cast<half*>({{name}})")
    cast_to_const_half_ptr_template = jinja2.Template(
        "reinterpret_cast<const half*>({{name}})"
    )
    header_src_template = jinja2.Template(
        """
#include <cuda_fp16.h>
{{extra_header}}
        """
    )

    half2_data_ref = ""
    dtype_to_cutlass_type: Dict[str, str] = field(
        default_factory=lambda: {
            "float16": "cutlass::half_t",
            "float": "float",
        }
    )

    def dtype_to_lib_type(self, dtype: str):
        return self.get_dtype_to_dtype(dtype, self.dtype_to_cutlass_type)
