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

#include <stdexcept>
#include <string>

#include <windows.h>

HMODULE SavedDllHandle;

BOOL WINAPI DllMain(
    HINSTANCE hinstDLL, // handle to DLL module
    DWORD fdwReason, // reason for calling function
    LPVOID lpvReserved) // reserved
{
  switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
      SavedDllHandle = hinstDLL;
      break;
  }
  return TRUE;
}

namespace ait {

#define TRIGGER_ERROR(message)                        \
  throw std::runtime_error(                           \
      (message) + " at file " + __FILE__ + ", line" + \
      std::to_string(__LINE__));

void GetConstantsBin(void** address, size_t* size) {
  HRSRC hResource = FindResource(SavedDllHandle, "constant_bin", "CUSTOMDATA");
  if (!hResource) {
    // Could not find a resource. Return zero values, because
    // linker won't include empty constant.bin file. So, this is an
    // expected behavior.
    *size = 0;
    *address = nullptr;
    return;
  }

  HGLOBAL hResourceData = LoadResource(SavedDllHandle, hResource);
  if (!hResourceData) {
    // could not load a resource
    auto errorCode = GetLastError();
    TRIGGER_ERROR(std::string(
        "LoadResource() call in GetConstantsBin() has failed with error " +
        std::to_string(errorCode)));
  }

  DWORD resourceSize = SizeofResource(SavedDllHandle, hResource);
  void* resourceData = LockResource(hResourceData);

  *size = resourceSize;
  *address = resourceData;
}

} // namespace ait
