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

from typing import Dict, List

import jinja2

from aitemplate.backend.target import Target

from aitemplate.compiler.ops.common.epilogue import FuncEnum


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
            "bfloat16": "bfloat16",
            "float32": "float",
            "float": "float",
            "int64": "int64_t",
            "int32": "int32_t",
        }
    )

    # find the size in bytes of a given backend type
    sizeof_types: Dict[str, int] = field(
        default_factory=lambda: {
            "uint8_t": 1,
            "half": 2,
            "bfloat16": 2,
            "float32": 4,
            "int64_t": 8,
            "int32_t": 4,
            "float": 4,
        }
    )

    # find a backend type for a given size in bytes
    # useful to find types 2 or 4 times larger than a given dtype
    # for vectorization purposes.
    type_for_size: Dict[int, str] = field(
        default_factory=lambda: {
            1: "uint8_t",
            2: "half",
            4: "float",
            8: "int64_t",
            16: "int4",
        }
    )

    backend_datatype_convertors: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "half": {"float": "__half2float"},
            "float": {"half": "__float2half_rn"},
        }
    )

    op_type_priority_list: List[str] = field(
        default_factory=lambda: [
            "half2",
            "half",
            "bfloat16_2",
            "bfloat16",
            "float",
        ]
    )
    func_enum_to_func_name: Dict[FuncEnum, Dict[str, str]] = field(
        default_factory=lambda: {
            FuncEnum.ADD: {
                "half2": "__hadd2",
                "bfloat16_2": "__hadd2",
                "half": "__hadd",
                "bfloat16": "__hadd",
                "float": "__fadd_rn",
            },
            FuncEnum.SUB: {
                "half2": "__hsub2",
                "bfloat16_2": "__hsub2",
                "half": "__hsub",
                "bfloat16": "__hsub",
                "float": "__fsub_rn",
            },
            FuncEnum.MUL: {
                "half2": "__hmul2",
                "bfloat16_2": "__hmul2",
                "half": "__hmul",
                "bfloat16": "__hmul",
                "float": "__fmul_rn",
            },
            FuncEnum.DIV: {
                "half2": "__h2div",
                "bfloat16_2": "__h2div",
                "half": "__hdiv",
                "bfloat16": "__hdiv",
                "float": "__fdiv_rn",
            },
            FuncEnum.COS: {
                "half2": "h2cos",
                "bfloat16_2": "h2cos",
                "half": "hcos",
                "bfloat16": "hcos",
                "float": "cosf",
            },
            FuncEnum.SIN: {
                "half2": "h2sin",
                "bfloat16_2": "h2sin",
                "half": "hsin" if Target.current().name() == "cuda" else "hsin_custom",
                "bfloat16": "hsin"
                if Target.current().name() == "cuda"
                else "hsin_custom",
                "float": "sinf",
            },
            FuncEnum.TANH: {
                "half2": "fast_tanh",
                "bfloat16_2": "fast_tanh",
                "half": "fast_tanh",
                "bfloat16": "fast_tanh",
                "float": "tanh",
            },
            FuncEnum.ABS: {
                "half2": "__habs2",
                "bfloat16_2": "__habs2",
                "half": "__habs",
                "bfloat16": "__habs",
                "float": "fabsf",
            },
            FuncEnum.LOGE: {
                "half2": "h2log",
                "bfloat16_2": "h2log",
                "half": "hlog",
                "bfloat16": "hlog",
                "float": "logf",
            },
            FuncEnum.EXP: {
                "half2": "h2exp",
                "bfloat16_2": "h2exp",
                "half": "hexp",
                "bfloat16": "hexp",
                "float": "expf",
            },
            FuncEnum.SQRT: {
                "half2": "h2sqrt",
                "bfloat16_2": "h2sqrt",
                "half": "hsqrt",
                "bfloat16": "hsqrt",
                "float": "sqrtf",
            },
            FuncEnum.MAX: {
                "half2": "hmax2_nan",
                "bfloat16_2": "hmax2_nan",
                "half": "hmax_nan",
                "bfloat16": "hmax_nan",
                "float": "fmaxf_nan",
            },
            FuncEnum.MIN: {
                "half2": "hmin2_nan",
                "bfloat16_2": "hmin2_nan",
                "half": "hmin_nan",
                "bfloat16": "hmin_nan",
                "float": "fminf_nan",
            },
            FuncEnum.SIGN: {
                "half2": "h2sign_custom",
                "bfloat16_2": "h2sign_custom",
                "half": "sign_custom<half>",
                "bfloat16": "sign_custom<bfloat16>",
                "float": "sign_custom<float>",
            },
            FuncEnum.SIGMOID: {
                "half2": "h2sigmoid_custom",
                "bfloat16_2": "h2sigmoid_custom",
                "half": "hsigmoid_custom",
                "bfloat16": "hsigmoid_custom",
                "float": "fsigmoid_custom",
            },
            FuncEnum.LRELU: {
                "half2": "leaky_relu",
                "bfloat16_2": "leaky_relu",
                "half": "leaky_relu",
                "bfloat16": "leaky_relu",
                "float": "leaky_relu",
            },
            FuncEnum.HARDTANH: {
                "half2": "h2hard_tanh",
                "bfloat16_2": "h2hard_tanh",
                "half": "hard_tanh",
                "float": "hard_tanh",
                "bfloat16": "hard_tanh",
            },
            FuncEnum.RELU: {
                "half2": "relu",
                "bfloat16_2": "relu",
                "half": "relu",
                "bfloat16": "relu",
                "float": "relu",
            },
            FuncEnum.NAN_TO_NUM: {
                "half2": "nan_to_num",
                "bfloat16_2": "nan_to_num",
                "half": "nan_to_num",
                "bfloat16": "nan_to_num",
                "float": "nan_to_num",
            },
            FuncEnum.CLAMP_NAN_TO_NUM: {
                "half2": "clamp_nan_to_num",
                "bfloat16_2": "clamp_nan_to_num",
                "half": "clamp_nan_to_num",
                "bfloat16": "clamp_nan_to_num",
                "float": "clamp_nan_to_num",
            },
            FuncEnum.SILU: {
                "half2": "h2silu",
                "bfloat16_2": "h2silu",
                "half": "hsilu",
                "bfloat16": "hsilu",
                "float": "fsilu",
            },
            FuncEnum.POW: {
                "half2": "h2pow",
                "bfloat16_2": "h2pow",
                "half": "hpow",
                "bfloat16": "hpow",
                "float": "fpow",
            },
            FuncEnum.GELU: {
                "half": "hgelu",
                "bfloat16": "hgelu",
                "float": "fgelu",
            },
            FuncEnum.FASTGELU: {
                "half": "h_fast_gelu",
                "bfloat16": "h_fast_gelu",
                "float": "f_fast_gelu",
            },
            FuncEnum.SOFTPLUS: {
                "half2": "h2softplus",
                "bfloat16_2": "h2softplus",
                "half": "hsoftplus",
                "bfloat16": "hsoftplus",
                "float": "fsoftplus",
            },
            FuncEnum.ELU: {
                "half2": "h2elu",
                "bfloat16_2": "h2elu",
                "half": "helu",
                "bfloat16": "helu",
                "float": "felu",
            },
            FuncEnum.SOFTSIGN: {
                "float": "fsoftsign",
                "half": "hsoftsign",
                "half2": "h2softsign",
                "bfloat16": "hsoftsign",
                "bfloat16_2": "h2softsign",
            },
            FuncEnum.FLOOR_DIV: {
                "float": "floor_div",
                "half": "floor_div",
                "half2": "floor_div",
                "bfloat16": "floor_div",
                "bfloat16_2": "floor_div",
            },
            FuncEnum.CELU: {
                "float": "fcelu",
                "half": "hcelu",
                "half2": "h2celu",
                "bfloat16": "hcelu",
                "bfloat16_2": "h2celu",
            },
        }
    )

    def get_elementwise_op_backend_type(
        self,
        num_elements: int,
        dtype: str,
    ) -> str:
        """
        Get a backend type execution in elementwise ops.
        For example, if we're dealing with fp16, we might be able to use half2 if num_elements is divisible by 2.
        """
        if dtype in ("float", "float32"):
            return "float"
        elif dtype == "float16":
            if num_elements % 2 == 0:
                return "half2"
            else:
                return "half"
        elif dtype == "bfloat16":
            if num_elements % 2 == 0:
                return "bfloat16_2"
            else:
                return "bfloat16"
        raise NotImplementedError("Unsupported dtype {}!".format(dtype))

    def get_elementwise_read_backend_type(
        self,
        num_elements: int,
        dtype: str,
    ) -> str:
        """
        Get a backend type for reading in elementwise ops.
        For example, if we're dealing with fp16 and num_elements is divisible by 8,
        we can use uint4.
        """
        if dtype in ("float", "float32"):
            num_elems_to_backend_type = ((4, "uint4"), (2, "uint2"), (1, "float"))

        elif dtype == "float16":
            num_elems_to_backend_type = (
                (8, "uint4"),
                (4, "uint2"),
                (2, "uint"),
                (1, "half"),
            )
        elif dtype == "bfloat16":
            num_elems_to_backend_type = (
                (8, "uint4"),
                (4, "uint2"),
                (2, "uint"),
                (1, "bfloat16"),
            )
        else:
            raise NotImplementedError("Unsupported dtype {}!".format(dtype))

        for mod, dtype in num_elems_to_backend_type:
            if num_elements % mod == 0:
                return dtype

        raise RuntimeError(
            f"Failed to infer data type due to invalid num elems to backend type mapping: {num_elems_to_backend_type}"
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

    cast_to_ptr_template = jinja2.Template("reinterpret_cast<{{dtype}}*>({{name}})")
    cast_to_half_ptr_template = jinja2.Template("reinterpret_cast<half*>({{name}})")
    cast_to_const_half_ptr_template = jinja2.Template(
        "reinterpret_cast<const half*>({{name}})"
    )
    header_src_template = jinja2.Template(
        """
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

using bfloat16 = hip_bfloat16;

{{extra_header}}
        """
    )
    half2_data_ref = ".data"

    dtype_to_ck_type: Dict[str, str] = field(
        default_factory=lambda: {
            "float16": "ck::half_t",
            "float32": "float",
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

    cast_to_ptr_template = jinja2.Template("reinterpret_cast<{{dtype}}*>({{name}})")
    cast_to_half_ptr_template = jinja2.Template("reinterpret_cast<half*>({{name}})")
    cast_to_const_half_ptr_template = jinja2.Template(
        "reinterpret_cast<const half*>({{name}})"
    )
    header_src_template = jinja2.Template(
        """
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using bfloat16 = nv_bfloat16;
using bfloat16_2 = nv_bfloat162;

{{extra_header}}
        """
    )

    half2_data_ref = ""
    dtype_to_cutlass_type: Dict[str, str] = field(
        default_factory=lambda: {
            "float16": "cutlass::half_t",
            "bfloat16": "cutlass::bfloat16_t",
            "float32": "float",
            "float": "float",
        }
    )

    def dtype_to_lib_type(self, dtype: str):
        return self.get_dtype_to_dtype(dtype, self.dtype_to_cutlass_type)
