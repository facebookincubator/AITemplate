// Single-Header version of printf.cu from the cuda-kat library
// implementing printf variants and string manipulation code
// See
// https://github.com/eyalroz/cuda-kat/blob/development/src/kat/on_device/c_standard_library/printf.cu
// copied from revision f771b5d5906d0f49e7500d32c2af91234c1cebad

/**
 * @author (c) Eyal Rozenberg <eyalroz1@gmx.com>
 *             2021-2022, Haifa, Palestine/Israel
 * @author (c) Marco Paland (info@paland.com)
 *             2014-2019, PALANDesign Hannover, Germany
 *
 * @note Others have made smaller contributions to this file: see the
 * contributors page at https://github.com/eyalroz/printf/graphs/contributors
 * or ask one of the authors. The original code for exponential specifiers was
 * contributed by Martijn Jasperse <m.jasperse@gmail.com>.
 *
 * @brief Small stand-alone implementation of the printf family of functions
 * (`(v)printf`, `(v)s(n)printf` etc., geared towards use on embedded systems
 * with a very limited resources.
 *
 * @note the implementations are thread-safe; re-entrant; use no functions from
 * the standard library; and do not dynamically allocate any memory.
 *
 * @license The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio> // for CUDA's builtin printf()

namespace kat {

/**
 * @brief Search for a character within a nul-terminated string.
 *
 * @param s The string to search
 * @param c A character value to search for
 * @return address of the first character with the value @p c
 * within string @p s, or nullptr if no character of @p s equals @p c .
 */
inline __device__ char* strchr(const char* s, int c) {
  const char* p = s;
  do {
    if (*p == static_cast<char>(c)) {
      return const_cast<char*>(p);
    }
  } while (*(p++) != '\0');
  return nullptr;
}

/**
 * @brief same as @ref std::strchr , except that the search begins
 * at the end of the string
 *
 * @note If @p c is '\0', it _will_ match the nul character
 * at the end of the string.
 *
 */
inline __device__ char* strrchr(const char* s, int c) {
  const char* last = nullptr;
  const char* p = s;
  do {
    if (*p == c) {
      last = p;
    }
  } while (*(p++) != '\0');
  return const_cast<char*>(last);
}

/**
 * An implementation of the C standard's snprintf/vsnprintf
 *
 * @param s An array in which to store the formatted string. It must be large
 * enough to fit either the entire formatted output, or at least @p count
 * characters. Alternatively, it can be `NULL`, in which case nothing will be
 * printed, and only the number of characters which _could_ have been printed is
 * tallied and returned.
 * @param n The maximum number of characters to write to the array, including a
 * terminating null character.
 * @param format A string specifying the format of the output, with %-marked
 * specifiers of how to interpret additional arguments.
 * @param arg Additional arguments to the function, one for each specifier in @p
 * format
 * @return The number of characters that COULD have been written into @p s, not
 * counting the terminating null character. A value equal or larger than @p
 * count indicates truncation. Only when the returned value is non-negative and
 * less than @p count, the null-terminated string has been fully and
 * successfully printed. If `nullptr` was passed as `s`, the number of
 * _intended_ characters will be returned without any characters being written
 * anywhere.
 */
__attribute__((device)) int snprintf(
    char* s,
    size_t count,
    const char* format,
    ...) __attribute__((format(__printf__, (3), (4))));
__attribute__((device)) int vsnprintf(
    char* s,
    size_t count,
    const char* format,
    va_list arg) __attribute__((format(__printf__, ((3)), (0))));

/**
 * An implementation of the C standard's printf/vprintf, via a self-allocated
 * buffer, backed by CUDA's `printf()`
 *
 * @note These functions will allocate some scratch memory to format a string
 * into, which will then be printed using CUDA's printf. This may be
 * inconvenient or dangerous, so **use of these function is _not_ recommended.**
 * Prefer @ref printf_with_scratch or @ref vprintf_with_scratch instead.
 *
 * @param format A string specifying the format of the output, with %-marked
 * specifiers of how to interpret additional arguments.
 * @param arg Additional arguments to the function, one for each %-specifier in
 * @p format string
 * @return The number of characters written to the output not counting the
 * terminating null character.
 */
__attribute__((device)) int printf(const char* format, ...)
    __attribute__((format(__printf__, (1), (2))));

__attribute__((device)) int vprintf(const char* format, va_list arg)
    __attribute__((format(__printf__, ((1)), (0))));

/**
 * An implementation of the C standard's printf/vprintf, backed by CUDA's
 * `printf()`, with a user-provided sized scratch buffer.
 *
 * @note These functions will not allocate anything on the heap.
 *
 * @param scratch an array for staging the formatted output before passing it to
 * CUDA's `printf()` function. The buffer must have at least @p count available
 * bytes. If `nullptr` is passed for `scratch`, nothing is written, but the
 * number of characters _to_ be written is returned.
 * @param count size of the @p scratch buffer
 * @param format A string specifying the format of the output, with %-marked
 * specifiers of how to interpret additional arguments.
 * @param arg additional arguments to the function, one for each %-specifier in
 * @p format string
 * @return The number of characters that COULD have been written into @p s, not
 * counting the terminating null character. A value equal or larger than @p
 * count indicates truncation. Only when the returned value is non-negative and
 * less than @p count, the null-terminated string has been fully and
 * successfully printed. If `nullptr` was passed as `s`, the number of
 * _intended_ characters will be returned without any characters being written
 * anywhere.
 */
__attribute__((device)) int vnprintf_with_scratch(
    char* scratch,
    size_t count,
    const char* format,
    va_list arg) __attribute__((format(__printf__, ((3)), (0))));
__attribute__((device)) int nprintf_with_scratch(
    char* scratch,
    size_t count,
    const char* format,
    ...) __attribute__((format(__printf__, (3), (4))));

/**
 * An implementation of the C standard's sprintf/vsprintf
 *
 * @note For security considerations (the potential for exceeding the buffer
 * bounds), please consider using the size-constrained variant, @ref
 * kat::snprintf / @ref kat::vsnprintf , instead.
 *
 * @param s An array in which to store the formatted string. It must be large
 * enough to fit the formatted output!
 * @param format A string specifying the format of the output, with %-marked
 * specifiers of how to interpret additional arguments.
 * @param arg Additional arguments to the function, one for each specifier in @p
 * format
 * @return The number of characters written into @p s, not counting the
 * terminating null character. If `nullptr` was passed as `s`, the number of
 * _intended_ characters will be returned without any characters being written
 * anywhere.
 */
__attribute__((device)) int sprintf(char* s, const char* format, ...)
    __attribute__((format(__printf__, (2), (3))));
__attribute__((device)) int vsprintf(char* s, const char* format, va_list arg)
    __attribute__((format(__printf__, ((2)), (0))));

} // namespace kat

// ---------------------------------------------------------------------------------------------------------------------
namespace kat {
namespace detail_ {
namespace printf {
enum {
  integer_buffer_size = 32,
  decimal_buffer_size = 32,
  default_float_precision = 6,
  num_decimal_digits_in_int64_t = 18,
  max_supported_precision = num_decimal_digits_in_int64_t - 1,
};
constexpr const double float_notation_threshold = 1e9;
namespace flags {
static_assert(sizeof(short) == 2, "Unexpected size of short");
static_assert(sizeof(int) == 4, "Unexpected size of int");
static_assert(sizeof(long) == 8, "Unexpected size of long");
enum : unsigned {
  zeropad = 1U << 0U,
  left = 1U << 1U,
  plus = 1U << 2U,
  space = 1U << 3U,
  hash = 1U << 4U,
  uppercase = 1U << 5U,
  char_ = 1U << 6U,
  short_ = 1U << 7U,
  int_ = 1U << 8U,
  long_ = 1U << 9U,
  long_long = 1U << 10U,
  precision = 1U << 11U,
  adapt_exp = 1U << 12U,
  pointer = 1U << 13U,
  signed_ = 1U << 14U,
  int8 = char_,
  int16 = short_,
  int32 = int_,
  int64 = long_
};
} // namespace flags
typedef unsigned int flags_t;
namespace base {
enum { binary = 2, octal = 8, decimal = 10, hex = 16 };
}
typedef uint8_t numeric_base_t;
typedef unsigned long long unsigned_value_t;
typedef long long signed_value_t;
typedef unsigned int printf_size_t;
enum { max_possible_buffer_size = 0x7fffffff };
namespace double_ {
static_assert(
    FLT_RADIX == 2,
    "Non-binary-radix floating-point types are unsupported.");
static_assert(DBL_MANT_DIG == 53, "Unsupported double type configuration");
typedef uint64_t uint_t;
enum {
  size_in_bits = 64,
  base_exponent = 1023,
  stored_mantissa_bits = DBL_MANT_DIG - 1,
};
enum : unsigned { exponent_mask = 0x7FFU };
union with_bit_access {
  uint_t U;
  double F;
  static __attribute__((device)) constexpr with_bit_access wrap(double x) {
    with_bit_access dwba = {.F = x};
    return dwba;
  }
  __attribute__((device)) constexpr __attribute__((device)) int exp2() const {
    return (int)((U >> stored_mantissa_bits) & exponent_mask) - base_exponent;
  }
};
struct components {
  int_fast64_t integral;
  int_fast64_t fractional;
  bool is_negative;
};
} // namespace double_
__attribute__((device)) static inline constexpr int get_sign_bit(double x) {
  return (
      int)(double_::with_bit_access::wrap(x).U >> (double_::size_in_bits - 1));
}
__attribute__((device)) static inline int get_exp2(double x) {
  return double_::with_bit_access::wrap(x).exp2();
}
template <typename T>
__attribute__((device)) constexpr T abs(T x) {
  return x > 0 ? x : -x;
}
template <typename T>
__attribute__((device)) constexpr unsigned_value_t abs_for_printing(T x) {
  return x > 0 ? x : -(signed_value_t)x;
}
typedef struct {
  void (*function)(char c, void* extra_arg);
  void* extra_function_arg;
  char* buffer;
  printf_size_t pos;
  printf_size_t max_chars;
} output_gadget_t;
__attribute__((noinline)) __attribute__((device)) static inline void
putchar_via_gadget(output_gadget_t* gadget, char c) {
  printf_size_t write_pos = gadget->pos++;
  if (write_pos >= gadget->max_chars) {
    return;
  }
  if (gadget->function != nullptr) {
    gadget->function(c, gadget->extra_function_arg);
  } else {
    gadget->buffer[write_pos] = c;
  }
}
__attribute__((device)) static inline void append_termination_with_gadget(
    output_gadget_t* gadget) {
  if (gadget->function != nullptr || gadget->max_chars == 0) {
    return;
  }
  if (gadget->buffer == nullptr) {
    return;
  }
  printf_size_t null_char_pos =
      gadget->pos < gadget->max_chars ? gadget->pos : gadget->max_chars - 1;
  gadget->buffer[null_char_pos] = '\0';
}
__attribute__((device)) static inline output_gadget_t discarding_gadget() {
  output_gadget_t gadget;
  gadget.function = nullptr;
  gadget.extra_function_arg = nullptr;
  gadget.buffer = nullptr;
  gadget.pos = 0;
  gadget.max_chars = 0;
  return gadget;
}
__attribute__((device)) static inline output_gadget_t buffer_gadget(
    char* buffer,
    size_t buffer_size) {
  printf_size_t usable_buffer_size = (buffer_size > max_possible_buffer_size)
      ? max_possible_buffer_size
      : (printf_size_t)buffer_size;
  output_gadget_t result = discarding_gadget();
  if (buffer != nullptr) {
    result.buffer = buffer;
    result.max_chars = usable_buffer_size;
  }
  return result;
}
__attribute__((device)) static inline printf_size_t strnlen_s_(
    const char* str,
    printf_size_t maxsize) {
  const char* s;
  for (s = str; *s && maxsize--; ++s)
    ;
  return (printf_size_t)(s - str);
}
__attribute__((device)) static inline constexpr bool is_digit_(char ch) {
  return (ch >= '0') && (ch <= '9');
}
__attribute__((device)) static printf_size_t atou_(const char** str) {
  printf_size_t i = 0U;
  while (is_digit_(**str)) {
    i = i * 10U + (printf_size_t)(*((*str)++) - '0');
  }
  return i;
}
__attribute__((device)) static void out_rev_(
    output_gadget_t* output,
    const char* buf,
    printf_size_t len,
    printf_size_t width,
    flags_t flags) {
  const printf_size_t start_pos = output->pos;
  if (!(flags & flags::left) && !(flags & flags::zeropad)) {
    for (printf_size_t i = len; i < width; i++) {
      putchar_via_gadget(output, ' ');
    }
  }
  while (len) {
    putchar_via_gadget(output, buf[--len]);
  }
  if (flags & flags::left) {
    while (output->pos - start_pos < width) {
      putchar_via_gadget(output, ' ');
    }
  }
}
__attribute__((device)) static void print_integer_finalization(
    output_gadget_t* __restrict__ output,
    char* __restrict__ buf,
    printf_size_t len,
    bool negative,
    numeric_base_t base,
    printf_size_t precision,
    printf_size_t width,
    flags_t flags) {
  printf_size_t unpadded_len = len;
  {
    if (!(flags & flags::left)) {
      if (width && (flags & flags::zeropad) &&
          (negative || (flags & (flags::plus | flags::space)))) {
        width--;
      }
      while ((flags & flags::zeropad) && (len < width) &&
             (len < detail_::printf::integer_buffer_size)) {
        buf[len++] = '0';
      }
    }
    while ((len < precision) && (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = '0';
    }
    if (base == base::octal && (len > unpadded_len)) {
      flags &= ~flags::hash;
    }
  }
  if (flags & (flags::hash | flags::pointer)) {
    if (!(flags & flags::precision) && len &&
        ((len == precision) || (len == width))) {
      if (unpadded_len < len) {
        len--;
      }
      if (len && (base == base::hex || base == base::binary) &&
          (unpadded_len < len)) {
        len--;
      }
    }
    if ((base == base::hex) && !(flags & flags::uppercase) &&
        (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = 'x';
    } else if (
        (base == base::hex) && (flags & flags::uppercase) &&
        (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = 'X';
    } else if (
        (base == base::binary) &&
        (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = 'b';
    }
    if (len < detail_::printf::integer_buffer_size) {
      buf[len++] = '0';
    }
  }
  if (len < detail_::printf::integer_buffer_size) {
    if (negative) {
      buf[len++] = '-';
    } else if (flags & flags::plus) {
      buf[len++] = '+';
    } else if (flags & flags::space) {
      buf[len++] = ' ';
    }
  }
  out_rev_(output, buf, len, width, flags);
}
__attribute__((device)) static void print_integer(
    output_gadget_t* output,
    unsigned_value_t value,
    bool negative,
    numeric_base_t base,
    printf_size_t precision,
    printf_size_t width,
    flags_t flags) {
  char buf[detail_::printf::integer_buffer_size];
  printf_size_t len = 0U;
  if (!value) {
    if (!(flags & flags::precision)) {
      buf[len++] = '0';
      flags &= ~flags::hash;
    } else if (base == base::hex) {
      flags &= ~flags::hash;
    }
  } else {
    do {
      const char digit = (char)(value % base);
      buf[len++] =
          (char)(digit < 10 ? '0' + digit : (flags & flags::uppercase ? 'A' : 'a') + digit - 10);
      value /= base;
    } while (value && (len < detail_::printf::integer_buffer_size));
  }
  print_integer_finalization(
      output, buf, len, negative, base, precision, width, flags);
}
__attribute__((device)) double power_of_10(int e) {
  switch (e) {
    case 0:
      return 1e00;
    case 1:
      return 1e01;
    case 2:
      return 1e02;
    case 3:
      return 1e03;
    case 4:
      return 1e04;
    case 5:
      return 1e05;
    case 6:
      return 1e06;
    case 7:
      return 1e07;
    case 8:
      return 1e08;
    case 9:
      return 1e09;
    case 10:
      return 1e10;
    case 11:
      return 1e11;
    case 12:
      return 1e12;
    case 13:
      return 1e13;
    case 14:
      return 1e14;
    case 15:
      return 1e15;
    case 16:
      return 1e16;
    case 17:
      return 1e17;
  }
  return 1;
}
__attribute__((device)) static double_::components get_components(
    double number,
    printf_size_t precision) {
  double_::components number_;
  number_.is_negative = get_sign_bit(number);
  double abs_number = (number_.is_negative) ? -number : number;
  number_.integral = (int_fast64_t)abs_number;
  double remainder =
      (abs_number - (double)number_.integral) * power_of_10((int)precision);
  number_.fractional = (int_fast64_t)remainder;
  remainder -= (double)number_.fractional;
  if (remainder > 0.5) {
    ++number_.fractional;
    if ((double)number_.fractional >= power_of_10((int)precision)) {
      number_.fractional = 0;
      ++number_.integral;
    }
  } else if (
      (remainder == 0.5) &&
      ((number_.fractional == 0U) || (number_.fractional & 1U))) {
    ++number_.fractional;
  }
  if (precision == 0U) {
    remainder = abs_number - (double)number_.integral;
    if ((!(remainder < 0.5) || (remainder > 0.5)) && (number_.integral & 1)) {
      ++number_.integral;
    }
  }
  return number_;
}
struct scaling_factor {
  double raw_factor;
  bool multiply;
};
__attribute__((device)) static double apply_scaling(
    double num,
    scaling_factor normalization) {
  return normalization.multiply ? num * normalization.raw_factor
                                : num / normalization.raw_factor;
}
__attribute__((device)) static double unapply_scaling(
    double normalized,
    scaling_factor normalization) {
  return normalization.multiply ? normalized / normalization.raw_factor
                                : normalized * normalization.raw_factor;
}
__attribute__((device)) static scaling_factor update_normalization(
    scaling_factor sf,
    double extra_multiplicative_factor) {
  scaling_factor result;
  int factor_exp2 = get_exp2(sf.raw_factor);
  int extra_factor_exp2 = get_exp2(extra_multiplicative_factor);
  if (abs(factor_exp2) > abs(extra_factor_exp2)) {
    result.multiply = false;
    result.raw_factor = sf.raw_factor / extra_multiplicative_factor;
  } else {
    result.multiply = true;
    result.raw_factor = extra_multiplicative_factor / sf.raw_factor;
  }
  return result;
}
__attribute__((device)) static double_::components get_normalized_components(
    bool negative,
    printf_size_t precision,
    double non_normalized,
    scaling_factor normalization,
    int floored_exp10) {
  double_::components components;
  components.is_negative = negative;
  double scaled = apply_scaling(non_normalized, normalization);
  bool close_to_representation_extremum =
      ((-floored_exp10 + (int)precision) >= DBL_MAX_10_EXP - 1);
  if (close_to_representation_extremum) {
    return get_components(negative ? -scaled : scaled, precision);
  }
  components.integral = (int_fast64_t)scaled;
  double remainder = non_normalized -
      unapply_scaling((double)components.integral, normalization);
  double prec_power_of_10 = power_of_10((int)precision);
  scaling_factor account_for_precision =
      update_normalization(normalization, prec_power_of_10);
  double scaled_remainder = apply_scaling(remainder, account_for_precision);
  double rounding_threshold = 0.5;
  components.fractional = (int_fast64_t)scaled_remainder;
  scaled_remainder -= (double)components.fractional;
  components.fractional += (scaled_remainder >= rounding_threshold);
  if (scaled_remainder == rounding_threshold) {
    components.fractional &= ~((int_fast64_t)0x1);
  }
  if ((double)components.fractional >= prec_power_of_10) {
    components.fractional = 0;
    ++components.integral;
  }
  return components;
}
__attribute__((device)) static void print_broken_up_decimal(
    double_::components number_,
    output_gadget_t* output,
    printf_size_t precision,
    printf_size_t width,
    flags_t flags,
    char* buf,
    printf_size_t len) {
  if (precision != 0U) {
    printf_size_t count = precision;
    if ((flags & flags::adapt_exp) && !(flags & flags::hash) &&
        (number_.fractional > 0)) {
      while (true) {
        int_fast64_t digit = number_.fractional % 10U;
        if (digit != 0) {
          break;
        }
        --count;
        number_.fractional /= 10U;
      }
    }
    if (number_.fractional > 0 || !(flags & flags::adapt_exp) ||
        (flags & flags::hash)) {
      while (len < decimal_buffer_size) {
        --count;
        buf[len++] = (char)('0' + number_.fractional % 10U);
        if (!(number_.fractional /= 10U)) {
          break;
        }
      }
      while ((len < decimal_buffer_size) && (count > 0U)) {
        buf[len++] = '0';
        --count;
      }
      if (len < decimal_buffer_size) {
        buf[len++] = '.';
      }
    }
  } else {
    if ((flags & flags::hash) && (len < decimal_buffer_size)) {
      buf[len++] = '.';
    }
  }
  while (len < decimal_buffer_size) {
    buf[len++] = (char)('0' + (number_.integral % 10));
    if (!(number_.integral /= 10)) {
      break;
    }
  }
  if (!(flags & flags::left) && (flags & flags::zeropad)) {
    if (width &&
        (number_.is_negative || (flags & (flags::plus | flags::space)))) {
      width--;
    }
    while ((len < width) && (len < decimal_buffer_size)) {
      buf[len++] = '0';
    }
  }
  if (len < decimal_buffer_size) {
    if (number_.is_negative) {
      buf[len++] = '-';
    } else if (flags & flags::plus) {
      buf[len++] = '+';
    } else if (flags & flags::space) {
      buf[len++] = ' ';
    }
  }
  out_rev_(output, buf, len, width, flags);
}
__attribute__((device)) static void print_decimal_number(
    output_gadget_t* __restrict__ output,
    double number,
    printf_size_t precision,
    printf_size_t width,
    flags_t flags,
    char* __restrict__ buf,
    printf_size_t len) {
  double_::components value_ = get_components(number, precision);
  print_broken_up_decimal(value_, output, precision, width, flags, buf, len);
}
__attribute__((device)) static void print_exponential_number(
    output_gadget_t* __restrict__ output,
    double number,
    printf_size_t precision,
    printf_size_t width,
    flags_t flags,
    char* __restrict__ buf,
    printf_size_t len) {
  const bool negative = get_sign_bit(number);
  double abs_number = negative ? -number : number;
  int floored_exp10;
  bool abs_exp10_covered_by_powers_table;
  scaling_factor normalization;
  if (abs_number == 0.0) {
    floored_exp10 = 0;
  } else {
    double exp10 = log10(abs_number);
    floored_exp10 = floor(exp10);
    double p10 = pow(10, floored_exp10);
    normalization.raw_factor = p10;
    abs_exp10_covered_by_powers_table = false;
  }
  bool fall_back_to_decimal_only_mode = false;
  if (flags & flags::adapt_exp) {
    int required_significant_digits = (precision == 0) ? 1 : (int)precision;
    fall_back_to_decimal_only_mode =
        (floored_exp10 >= -4 && floored_exp10 < required_significant_digits);
    int precision_ = fall_back_to_decimal_only_mode
        ? (int)precision - 1 - floored_exp10
        : (int)precision - 1;
    precision = (precision_ > 0 ? (unsigned)precision_ : 0U);
    flags |= flags::precision;
  }
  normalization.multiply =
      (floored_exp10 < 0 && abs_exp10_covered_by_powers_table);
  bool should_skip_normalization =
      (fall_back_to_decimal_only_mode || floored_exp10 == 0);
  double_::components decimal_part_components = should_skip_normalization
      ? get_components(negative ? -abs_number : abs_number, precision)
      : get_normalized_components(
            negative, precision, abs_number, normalization, floored_exp10);
  if (fall_back_to_decimal_only_mode) {
    if ((flags & flags::adapt_exp) && floored_exp10 >= -1 &&
        decimal_part_components.integral == power_of_10(floored_exp10 + 1)) {
      floored_exp10++;
      precision--;
    }
  } else {
    if (decimal_part_components.integral >= 10) {
      floored_exp10++;
      decimal_part_components.integral = 1;
      decimal_part_components.fractional = 0;
    }
  }
  printf_size_t exp10_part_width = fall_back_to_decimal_only_mode ? 0U
      : (abs(floored_exp10) < 100)                                ? 4U
                                                                  : 5U;
  printf_size_t decimal_part_width = ((flags & flags::left) && exp10_part_width)
      ? 0U
      : ((width > exp10_part_width) ? width - exp10_part_width : 0U);
  const printf_size_t printed_exponential_start_pos = output->pos;
  print_broken_up_decimal(
      decimal_part_components,
      output,
      precision,
      decimal_part_width,
      flags,
      buf,
      len);
  if (!fall_back_to_decimal_only_mode) {
    putchar_via_gadget(output, (flags & flags::uppercase) ? 'E' : 'e');
    print_integer(
        output,
        abs_for_printing(floored_exp10),
        floored_exp10 < 0,
        10,
        0,
        exp10_part_width - 1,
        flags::zeropad | flags::plus);
    if (flags & flags::left) {
      while (output->pos - printed_exponential_start_pos < width) {
        putchar_via_gadget(output, ' ');
      }
    }
  }
}
__attribute__((device)) static void print_floating_point(
    output_gadget_t* output,
    double value,
    printf_size_t precision,
    printf_size_t width,
    flags_t flags,
    bool prefer_exponential) {
  char buf[decimal_buffer_size];
  printf_size_t len = 0U;
  if (value != value) {
    out_rev_(output, "nan", 3, width, flags);
    return;
  }
  if (value < -DBL_MAX) {
    out_rev_(output, "fni-", 4, width, flags);
    return;
  }
  if (value > DBL_MAX) {
    out_rev_(
        output,
        (flags & flags::plus) ? "fni+" : "fni",
        (flags & flags::plus) ? 4U : 3U,
        width,
        flags);
    return;
  }
  if (!prefer_exponential &&
      ((value > float_notation_threshold) ||
       (value < -float_notation_threshold))) {
    print_exponential_number(output, value, precision, width, flags, buf, len);
    return;
  }
  if (!(flags & flags::precision)) {
    precision = default_float_precision;
  }
  while ((len < decimal_buffer_size) && (precision > max_supported_precision)) {
    buf[len++] = '0';
    precision--;
  }
  if (prefer_exponential)
    print_exponential_number(output, value, precision, width, flags, buf, len);
  else
    print_decimal_number(output, value, precision, width, flags, buf, len);
}
__attribute__((device)) static flags_t parse_flags(const char** format) {
  flags_t flags = 0U;
  do {
    switch (**format) {
      case '0':
        flags |= flags::zeropad;
        (*format)++;
        break;
      case '-':
        flags |= flags::left;
        (*format)++;
        break;
      case '+':
        flags |= flags::plus;
        (*format)++;
        break;
      case ' ':
        flags |= flags::space;
        (*format)++;
        break;
      case '#':
        flags |= flags::hash;
        (*format)++;
        break;
      default:
        return flags;
    }
  } while (true);
}
__attribute__((device)) static int vsnprintf(
    output_gadget_t* output,
    const char* format,
    va_list args) {
  while (*format) {
    if (*format != '%') {
      putchar_via_gadget(output, *format);
      format++;
      continue;
    } else {
      format++;
    }
    flags_t flags = parse_flags(&format);
    printf_size_t width = 0U;
    if (is_digit_(*format)) {
      width = (printf_size_t)atou_(&format);
    } else if (*format == '*') {
      const int w = __builtin_va_arg(args, int);
      if (w < 0) {
        flags |= flags::left;
        width = (printf_size_t)-w;
      } else {
        width = (printf_size_t)w;
      }
      format++;
    }
    printf_size_t precision = 0U;
    if (*format == '.') {
      flags |= flags::precision;
      format++;
      if (is_digit_(*format)) {
        precision = (printf_size_t)atou_(&format);
      } else if (*format == '*') {
        const int precision_ = __builtin_va_arg(args, int);
        precision = precision_ > 0 ? (printf_size_t)precision_ : 0U;
        format++;
      }
    }
    switch (*format) {
      case 'I': {
        format++;
        switch (*format) {
          case '8':
            flags |= flags::int8;
            format++;
            break;
          case '1':
            format++;
            if (*format == '6') {
              format++;
              flags |= flags::int16;
            }
            break;
          case '3':
            format++;
            if (*format == '2') {
              format++;
              flags |= flags::int32;
            }
            break;
          case '6':
            format++;
            if (*format == '4') {
              format++;
              flags |= flags::int64;
            }
            break;
          default:
            break;
        }
        break;
      }
      case 'l':
        flags |= flags::long_;
        format++;
        if (*format == 'l') {
          flags |= flags::long_long;
          format++;
        }
        break;
      case 'h':
        flags |= flags::short_;
        format++;
        if (*format == 'h') {
          flags |= flags::char_;
          format++;
        }
        break;
      case 't':
      case 'j':
      case 'z':
        static_assert(
            sizeof(ptrdiff_t) == sizeof(long), "Unexpected sizeof(ptrdiff_t)");
        static_assert(
            sizeof(intmax_t) == sizeof(long), "Unexpected sizeof(intmax_t)");
        static_assert(
            sizeof(size_t) == sizeof(long), "Unexpected sizeof(size_t)");
        flags |= flags::long_;
        format++;
        break;
      default:
        break;
    }
    switch (*format) {
      case 'd':
      case 'i':
      case 'u':
      case 'x':
      case 'X':
      case 'o':
      case 'b': {
        if (*format == 'd' || *format == 'i') {
          flags |= flags::signed_;
        }
        numeric_base_t base;
        if (*format == 'x' || *format == 'X') {
          base = base::hex;
        } else if (*format == 'o') {
          base = base::octal;
        } else if (*format == 'b') {
          base = base::binary;
        } else {
          base = base::decimal;
          flags &= ~flags::hash;
        }
        if (*format == 'X') {
          flags |= flags::uppercase;
        }
        format++;
        if (flags & flags::precision) {
          flags &= ~flags::zeropad;
        }
        if (flags & flags::signed_) {
          if (flags & flags::long_long) {
            const long long value = __builtin_va_arg(args, long long);
            print_integer(
                output,
                abs_for_printing(value),
                value < 0,
                base,
                precision,
                width,
                flags);
          } else if (flags & flags::long_) {
            const long value = __builtin_va_arg(args, long);
            print_integer(
                output,
                abs_for_printing(value),
                value < 0,
                base,
                precision,
                width,
                flags);
          } else {
            const int value = (flags & flags::char_)
                ? (signed char)__builtin_va_arg(args, int)
                : (flags & flags::short_)
                ? (short int)__builtin_va_arg(args, int)
                : __builtin_va_arg(args, int);
            print_integer(
                output,
                abs_for_printing(value),
                value < 0,
                base,
                precision,
                width,
                flags);
          }
        } else {
          flags &= ~(flags::plus | flags::space);
          if (flags & flags::long_long) {
            print_integer(
                output,
                (unsigned_value_t) __builtin_va_arg(args, unsigned long long),
                false,
                base,
                precision,
                width,
                flags);
          } else if (flags & flags::long_) {
            print_integer(
                output,
                (unsigned_value_t) __builtin_va_arg(args, unsigned long),
                false,
                base,
                precision,
                width,
                flags);
          } else {
            const unsigned int value = (flags & flags::char_)
                ? (unsigned char)__builtin_va_arg(args, unsigned int)
                : (flags & flags::short_)
                ? (unsigned short int)__builtin_va_arg(args, unsigned int)
                : __builtin_va_arg(args, unsigned int);
            print_integer(
                output,
                (unsigned_value_t)value,
                false,
                base,
                precision,
                width,
                flags);
          }
        }
        break;
      }
        enum : bool { prefer_decimal = false, prefer_exponential = true };
      case 'f':
      case 'F':
        if (*format == 'F')
          flags |= flags::uppercase;
        print_floating_point(
            output,
            __builtin_va_arg(args, double),
            precision,
            width,
            flags,
            prefer_decimal);
        format++;
        break;
      case 'e':
      case 'E':
      case 'g':
      case 'G':
        if ((*format == 'g') || (*format == 'G'))
          flags |= flags::adapt_exp;
        if ((*format == 'E') || (*format == 'G'))
          flags |= flags::uppercase;
        print_floating_point(
            output,
            __builtin_va_arg(args, double),
            precision,
            width,
            flags,
            prefer_exponential);
        format++;
        break;
      case 'c': {
        printf_size_t l = 1U;
        if (!(flags & flags::left)) {
          while (l++ < width) {
            putchar_via_gadget(output, ' ');
          }
        }
        putchar_via_gadget(output, (char)__builtin_va_arg(args, int));
        if (flags & flags::left) {
          while (l++ < width) {
            putchar_via_gadget(output, ' ');
          }
        }
        format++;
        break;
      }
      case 's': {
        const char* p = __builtin_va_arg(args, char*);
        if (p == nullptr) {
          out_rev_(output, ")llun(", 6, width, flags);
        } else {
          printf_size_t l =
              strnlen_s_(p, precision ? precision : max_possible_buffer_size);
          if (flags & flags::precision) {
            l = (l < precision ? l : precision);
          }
          if (!(flags & flags::left)) {
            while (l++ < width) {
              putchar_via_gadget(output, ' ');
            }
          }
          while ((*p != 0) && (!(flags & flags::precision) || precision)) {
            putchar_via_gadget(output, *(p++));
            --precision;
          }
          if (flags & flags::left) {
            while (l++ < width) {
              putchar_via_gadget(output, ' ');
            }
          }
        }
        format++;
        break;
      }
      case 'p': {
        width = sizeof(void*) * 2U + 2;
        flags |= flags::zeropad | flags::pointer;
        uintptr_t value = (uintptr_t) __builtin_va_arg(args, void*);
        (value == (uintptr_t) nullptr)
            ? out_rev_(output, ")lin(", 5, width, flags)
            : print_integer(
                  output,
                  (unsigned_value_t)value,
                  false,
                  base::hex,
                  precision,
                  width,
                  flags);
        format++;
        break;
      }
      case '%':
        putchar_via_gadget(output, '%');
        format++;
        break;
      case 'n': {
        if (flags & flags::char_)
          *(__builtin_va_arg(args, char*)) = (char)output->pos;
        else if (flags & flags::short_)
          *(__builtin_va_arg(args, short*)) = (short)output->pos;
        else if (flags & flags::long_)
          *(__builtin_va_arg(args, long*)) = (long)output->pos;
        else if (flags & flags::long_long)
          *(__builtin_va_arg(args, long long*)) = (long long int)output->pos;
        else
          *(__builtin_va_arg(args, int*)) = (int)output->pos;
        format++;
        break;
      }
      default:
        putchar_via_gadget(output, *format);
        format++;
        break;
    }
  }
  append_termination_with_gadget(output);
  return (int)output->pos;
}
} // namespace printf
} // namespace detail_
__attribute__((device)) int vprintf(const char* format, va_list arg) {
  detail_::printf::output_gadget_t gadget =
      detail_::printf::discarding_gadget();
  int ret = vsnprintf(&gadget, format, arg);
  if (ret < 0) {
    return ret;
  }
  size_t count = ret + 1;
  char* scratch = (char*)malloc(count);
  if (scratch == nullptr) {
    return -1;
  }
  ret = vsnprintf(scratch, count, format, arg);
  if (ret < 0) {
    free(scratch);
    return ret;
  }
  ret = printf("%s", scratch);
}
__attribute__((device)) int vsnprintf(
    char* s,
    size_t n,
    const char* format,
    va_list arg) {
  detail_::printf::output_gadget_t gadget =
      detail_::printf::buffer_gadget(s, n);
  return detail_::printf::vsnprintf(&gadget, format, arg);
}
__attribute__((device)) int vsprintf(char* s, const char* format, va_list arg) {
  return vsnprintf(s, detail_::printf::max_possible_buffer_size, format, arg);
}
__attribute__((device)) inline int vnprintf_with_scratch(
    char* scratch,
    size_t count,
    const char* format,
    va_list arg) {
  const int ret = vsnprintf(scratch, count, format, arg);
  if (scratch == nullptr) {
    return ret;
  }
  if (ret > 0) {
    return printf("%s", scratch);
  }
};
__attribute__((device)) int printf(const char* format, ...) {
  va_list args;
  __builtin_va_start(args, format);
  const int ret = vprintf(format, args);
  __builtin_va_end(args);
  return ret;
}
__attribute__((device)) int sprintf(char* s, const char* format, ...) {
  va_list args;
  __builtin_va_start(args, format);
  const int ret = vsprintf(s, format, args);
  __builtin_va_end(args);
  return ret;
}
__attribute__((device)) int snprintf(
    char* s,
    size_t n,
    const char* format,
    ...) {
  va_list args;
  __builtin_va_start(args, format);
  const int ret = vsnprintf(s, n, format, args);
  __builtin_va_end(args);
  return ret;
}
__attribute__((device)) int nprintf_with_scratch(
    char* scratch_buffer,
    size_t count,
    const char* format,
    ...) {
  va_list args;
  __builtin_va_start(args, format);
  return vnprintf_with_scratch(scratch_buffer, count, format, args);
}
} // namespace kat
