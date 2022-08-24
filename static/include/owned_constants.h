#pragma once
// This exposes the raw data for constants that are
// compiled into the final .so. When a ModelContainer is created,
// it copies this data into some owned GPU memory.

#include <array>
#include <cstdint>
#include <utility>

namespace ait {

struct ConstantInfo {
  // Unowned pointer w/ static lifetime
  const char* name;
  // Offset into _binary_constants_bin_start
  size_t data_offset;
  // How big is this tensor in bytes?
  size_t num_bytes;
};

} // namespace ait

// At codegen time, we write out a binary file called constants.bin.
// We then turn the raw binary to an object file that exposes this
// symbol and link it into the final .so.
// For information on the binary format, see `man objcopy`, under
// the "binary-architecture" flag:
// https://man7.org/linux/man-pages/man1/objcopy.1.html
extern const uint8_t _binary_constants_bin_start[];
extern const uint8_t _binary_constants_bin_end[];
