//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
#pragma once

namespace ait {

// This structure is used to pack the offset metadata related to a
// jagged Tensor's first dimension: JaggedIntVar. The offsets are not
// available in compile time, as they are coming in a rank-1 Tensor.
// In runtime, the members of the structure are set by the make_jagged
// op's back-end, from the corresponding rank-1 offset Tensors' length
// and data. The OFFSET_TYPE can be either int32 or int64. The number
// of offset arrays is known in compile time, hence specified as the
// NUM_OFFSET_ARRAYS template argument here.
template <typename OFFSET_TYPE, int32_t NUM_OFFSET_ARRAYS>
struct JaggedOffsets {
  // the lengths the individual offset arrays
  int64_t lengths[NUM_OFFSET_ARRAYS]{0};
  // the data in each of the offset arrays
  // (i.e., the offsets of the JaggedIntVar)
  const OFFSET_TYPE* data[NUM_OFFSET_ARRAYS]{nullptr};
};

} // namespace ait
