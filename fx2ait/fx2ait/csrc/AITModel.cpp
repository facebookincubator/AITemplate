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
#include "AITModel.h"

#include "picojson.h"

namespace torch::aitemplate {
// const string for serialization
const static std::string LIB_BASENAME_STR = "library_basename";
const static std::string INPUT_NAMES_STR = "input_names";
const static std::string OUTPUT_NAMES_STR = "output_names";
const static std::string FLOATING_POINT_INPUT_DTYPE_STR =
    "floating_point_input_dtype";
const static std::string FLOATING_POINT_OUTPUT_DTYPE_STR =
    "floating_point_output_dtype";
std::string AITModel::serialize() const {
  std::string result;
  picojson::object var;
  picojson::array pick_input_names;
  var[LIB_BASENAME_STR] = picojson::value(aitModelImpl_.libraryBasename());
  for (const auto& entry : aitModelImpl_.inputNames()) {
    pick_input_names.push_back(picojson::value(entry));
  }
  var[INPUT_NAMES_STR] = picojson::value(pick_input_names);
  picojson::array pick_output_names;
  for (const auto& entry : aitModelImpl_.outputNames()) {
    pick_output_names.push_back(picojson::value(entry));
  }
  var[OUTPUT_NAMES_STR] = picojson::value(pick_output_names);
  var[FLOATING_POINT_INPUT_DTYPE_STR] = picojson::value(std::to_string(
      static_cast<int16_t>(aitModelImpl_.floatingPointInputDtype().value())));

  var[FLOATING_POINT_OUTPUT_DTYPE_STR] = picojson::value(std::to_string(
      static_cast<int16_t>(aitModelImpl_.floatingPointOutputDtype().value())));

  result = picojson::value(var).serialize();
  return result;
}

void AITModel::loadAsTorchClass() {
  // Calling this function will make sure that the static content of this file
  // will be executed. I.e. the most important part here is registering the
  // AITModel class for Python environment (i.e. torch::deploy).
  LOG(INFO) << "Making sure AITModel is registered via torch::class_";
}

static auto registerAITModel =
    torch::class_<AITModel>("ait", "AITModel")
        .def(torch::init<
             std::string,
             std::vector<std::string>,
             std::vector<std::string>,
             c10::optional<at::ScalarType>,
             c10::optional<at::ScalarType>,
             int64_t>())
        .def("forward", &AITModel::forward)
        .def("profile", &AITModel::profile)
        .def("get_library_path", &AITModel::libraryPath)
        .def_property(
            "use_cuda_graph",
            &AITModel::getUseCudaGraph,
            &AITModel::setUseCudaGraph)
        .def_static(
            "register_library_name_to_path_map",
            [](c10::Dict<std::string, std::string> dict) {
              std::unordered_map<std::string, std::string> map;
              for (const auto& entry : dict) {
                map[entry.key()] = entry.value();
              }
              AITModelImpl::registerLibraryNameToPathMap(std::move(map));
            })
        .def_pickle(
            [](const c10::intrusive_ptr<AITModel>& self) -> std::string {
              return self->serialize();
            },
            [](const std::string& data) {
              picojson::value var;
              const char* json = data.c_str();
              picojson::parse(var, json, json + strlen(json));
              std::vector<std::string> input_names;
              for (const auto name :
                   var.get(INPUT_NAMES_STR).get<picojson::array>()) {
                input_names.push_back(name.get<std::string>());
              }
              std::vector<std::string> output_names;
              for (const auto name :
                   var.get(OUTPUT_NAMES_STR).get<picojson::array>()) {
                output_names.push_back(name.get<std::string>());
              }
              auto floating_point_input_dtype =
                  std::stoi(var.get(FLOATING_POINT_INPUT_DTYPE_STR)
                                .get<std::string>()
                                .c_str());
              auto floating_point_output_dtype =
                  std::stoi(var.get(FLOATING_POINT_OUTPUT_DTYPE_STR)
                                .get<std::string>()
                                .c_str());
              return c10::make_intrusive<AITModel>(
                  AITModelImpl::getFullPathForLibraryName(
                      var.get(LIB_BASENAME_STR).get<std::string>().c_str()),
                  input_names,
                  output_names,
                  static_cast<at::ScalarType>(floating_point_input_dtype),
                  static_cast<at::ScalarType>(floating_point_output_dtype));
            });
} // namespace torch::aitemplate
