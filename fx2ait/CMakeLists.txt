cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(ait_model)
find_package(Torch REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/picojson
)

# Define our library target
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../static/include)
add_library(ait_model SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/fx2ait/csrc/AITModel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fx2ait/csrc/AITModelImpl.cpp
)

# Link against LibTorch
target_link_libraries(ait_model "${TORCH_LIBRARIES}")
