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
A variance kernel implementation based on the Welford's onlien algorithm:

https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

"""

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.reduce import reduce_3d


EXTRA_CODE_TEMPLATE = jinja2.Template(
    """
namespace {

template <typename ElementT, bool BesselCorrection>
struct WelfordData {
  int32_t count;
  ElementT mean;
  ElementT m2;

  CUTLASS_HOST_DEVICE
  WelfordData() : count(0), mean(0), m2(0) {}

  CUTLASS_HOST_DEVICE
  WelfordData(ElementT mean_) : count(1), mean(mean_), m2(0) {}

  CUTLASS_HOST_DEVICE
  WelfordData(int) : count(0), mean(0), m2(0) {}

  CUTLASS_HOST_DEVICE
  WelfordData(int count_, ElementT mean_, ElementT m2_)
    : count(count_), mean(mean_), m2(m2_) {}

  CUTLASS_HOST_DEVICE
  WelfordData reduce(WelfordData existing, WelfordData new_data) {
    if (new_data.count == 0) {
        return existing;
    }
    if (existing.count == 0 && new_data.count > 0) {
        return new_data;
    }
    ElementT mean = existing.mean;
    ElementT m2 = existing.m2;

    // combine two results
    if (count > 0 && new_data.count > 0) {
      ElementT delta = new_data.mean - mean;
      int new_count = new_data.count + count;
      ElementT nb_over_n = ElementT(new_data.count) / ElementT(new_count);
      mean = mean + delta * nb_over_n;
      m2 =  m2 + new_data.m2 + delta * delta * nb_over_n * ElementT(count);
      return WelfordData(new_count, mean, m2);
    }

    count++;
    ElementT delta = new_data.mean - mean;
    mean = mean + delta / ElementT(count);
    ElementT delta2 = new_data.mean - mean;
    m2 += delta * delta2;
    return WelfordData(count, mean, m2);
  }
};

} // anonymous namespace

namespace cutlass {

namespace arch {

/// note that we add this extra load utility for loading our WelfordData, which
/// requires more bytes to be loaded with kReduceVector = 4.
///
/// ld.shared - 2 of 128b
template <>
CUTLASS_DEVICE
void shared_load<32>(void *dst, uint32_t ptr) {
  uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
  asm volatile("ld.shared.v4.u32 {{ '{%0, %1, %2, %3}, [%4]' }};\\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));

  dst_u128++;
  ptr = ptr + sizeof(uint4);
  asm volatile("ld.shared.v4.u32 {{ '{%0, %1, %2, %3}, [%4]' }};\\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));
}

template <>
CUTLASS_DEVICE
void shared_load<48>(void *dst, uint32_t ptr) {
  uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
  asm volatile("ld.shared.v4.u32 {{ '{%0, %1, %2, %3}, [%4]' }};\\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));

  dst_u128++;
  ptr = ptr + sizeof(uint4);
  asm volatile("ld.shared.v4.u32 {{ '{%0, %1, %2, %3}, [%4]' }};\\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));

  dst_u128++;
  ptr = ptr + sizeof(uint4);
  asm volatile("ld.shared.v4.u32 {{ '{%0, %1, %2, %3}, [%4]' }};\\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));
}

} // namespace arch

template <typename ElementT, bool BesselCorrection>
struct NumericConverter<WelfordData<ElementT, BesselCorrection>,
                        ElementT,
                        FloatRoundStyle::round_to_nearest> {

  using result_type = WelfordData<ElementT, BesselCorrection>;
  using source_type = ElementT;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return WelfordData<ElementT, BesselCorrection>(-1, static_cast<ElementT>(s), ElementT(0));
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <typename ElementT, bool BesselCorrection>
struct NumericConverter<ElementT,
                        WelfordData<ElementT, BesselCorrection>,
                        FloatRoundStyle::round_to_nearest> {

  using result_type = ElementT;
  using source_type = WelfordData<ElementT, BesselCorrection>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    if (BesselCorrection) {
      // Bessel's correction (unbias = true)
      if (s.count <= 1) {
        return ElementT(nanf("Not a Number"));
      } else {
        return s.m2 / ElementT((int)(s.count - 1));
      }
    } else {
      // sample variance
      if (s.count <= 0) {
        return ElementT(nanf("Not a Number"));
      } else {
        return s.m2 / ElementT((int)(s.count));
      }
    }
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <typename T>
struct welford_op {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    return lhs.reduce(lhs, rhs);
  }
};

template <typename T, int N>
struct welford_op<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    welford_op<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    welford_op<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    welford_op<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

} // namespace cutlass
"""
)


@registry.reg("cuda.var.func_decl")
def var_gen_function_decl(func_attrs) -> str:
    """the registered function for generating var function declaration

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds the attributes of this var op

    Returns
    -------
    str
        returns the rendered function declaration with appropriate replacements
    """
    return reduce_3d.gen_function_decl(func_attrs)


@registry.reg("cuda.var.gen_function")
def var_gen_function(func_attrs) -> str:
    """the registered function for generating var kernel and all of
    its auxiliary functions

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds the attributes of this var op

    Returns
    -------
    str
        returns the rendered code for the complete implementation of this var op
    """
    bessel = "true" if func_attrs["unbiased"] else "false"
    backend_spec = CUDASpec()
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    acc_type = f"WelfordData<{elem_output_type}, {bessel}>"
    return reduce_3d.gen_function(
        func_attrs,
        "cutlass::welford_op",
        reduce_3d.DEFAULT_PROLOGUE_TEMPLATE,
        reduce_3d.DEFAULT_EPILOGUE_SCALAR_TEMPLATE,
        EXTRA_CODE_TEMPLATE.render(),
        accumulation_type=acc_type,
    )


@registry.reg("cuda.var.func_call")
def var_gen_function_call(func_attrs, indent="  "):
    """the registered function for generating a function call to var

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        holds the attributes of this var op
    indent : str, optional
        indentation for each line of the rendered code (default "  ")

    Returns
    -------
    str
        returns rendered code for invoking the reduce op
    """
    return reduce_3d.gen_function_call(func_attrs, indent)
