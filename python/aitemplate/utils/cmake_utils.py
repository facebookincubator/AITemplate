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
CMake utils
"""

import struct
from pathlib import Path


def _binarize_uint64(stream_in, stream_out) -> int:
    CHUNK_SIZE = 4096

    data = stream_in.read(CHUNK_SIZE)
    global_offset = 0
    while data:
        data_len = len(data)
        data_len8 = (data_len) // 8 * 8

        offset = 0
        while offset < data_len8:
            if global_offset != 0:
                stream_out.write(",")
            unpacked = struct.unpack_from("<Q", data, offset)
            stream_out.write(hex(unpacked[0]))
            # # An alternative version, maybe faster, maybe not
            # stream_out.write(data[offset:offset + 8][::-1].hex())

            offset += 8
            global_offset += 8
            if global_offset % 64 == 0:
                stream_out.write("\n")

        # leftovers
        if offset < data_len:
            leftover = 0
            multiplier = 1
            if global_offset != 0:
                stream_out.write(",")

            while offset < data_len:
                leftover += multiplier * struct.unpack_from("<B", data, offset)[0]
                offset += 1
                global_offset += 1
                multiplier *= 256

            stream_out.write(hex(leftover))

        # done
        data = stream_in.read(CHUNK_SIZE)

    return global_offset


def constants_bin_2_cpp(constants_bin_filename: Path, constants_cpp_filename: Path):
    with constants_bin_filename.open("rb") as stream_in:
        with constants_cpp_filename.open("w") as stream_out:
            stream_out.write("#include <cstdint>\n\n")
            stream_out.write("constexpr uint64_t data[] = {")
            data_size = _binarize_uint64(stream_in, stream_out)
            stream_out.write("};\n\n")
            stream_out.write(
                "const uint8_t* _binary_constants_bin_start = reinterpret_cast<const uint8_t*>(data) + 0;\n"
            )
            stream_out.write(
                f"const uint8_t* _binary_constants_bin_end = reinterpret_cast<const uint8_t*>(data) + {data_size};\n"
            )
