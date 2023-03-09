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
nms kernel codegen for CUDA.
"""

import os
from typing import Any, Dict

import jinja2

from aitemplate.backend.common.vision_ops.efficient_nms_kernel import kernel

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{kernel}}

}  // namespace

{{func_signature}}
{

    const int N = *batch;
    const int R = *num_rois;
    const int C = *num_classes;

    EfficientNMSParameters mParam;
    mParam.iouThreshold = iouThreshold;
    mParam.scoreThreshold = 0.001;
    mParam.boxDecoder = false;
    mParam.numOutputBoxesPerClass = nmsMaxOut;
    mParam.numOutputBoxes = nmsMaxOut;
    mParam.batchSize = N;
    mParam.numBoxElements = R * C * 4;
    mParam.numScoreElements = R * C;
    mParam.numAnchors = R;
    mParam.numClasses = C;
    mParam.shareLocation = (C == 1) ? true : false;
    mParam.outputONNXIndices = false;
    mParam.scoreSigmoid = false;
    mParam.numSelectedBoxes = 5000;

    const void* const boxesInput = proposals;
    const void* const scoresInput = fgScores;
    const void* const anchorsInput = nullptr;

    void* numDetectionsOutput = num_detections;
    void* nmsBoxesOutput = detection_boxes;
    void* nmsScoresOutput = detection_scores;
    void* nmsClassesOutput = detection_classe;

    return EfficientNMSInference(mParam, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
        nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, nullptr, workspace, stream);


}
    """
)

PROFILER_TEMPLATE = jinja2.Template(
    """
#include <iostream>
{{header_files}}
size_t GLOBAL_WORKSPACE_SIZE = 0;

namespace {

{{kernel}}

}  // namespace

int main(int argc, char** argv) {
  float runtime_ms = 0;
  int batchSize = std::stoi(argv[1]);
  int numScoreElements = std::stoi(argv[2]);
  int numClasses = std::stoi(argv[3]);
  GLOBAL_WORKSPACE_SIZE = EfficientNMSWorkspaceSize<{{elem_input_type}}>(batchSize, numScoreElements, numClasses);

  std::cout << "TIME:" << runtime_ms << std::endl;
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* num_detections,
                   void* detection_boxes,
                   void* detection_scores,
                   void* detection_classe,
                   const void* proposals,
                   const void* fgScores,
                   int64_t* batch,
                   int64_t* num_rois,
                   int64_t* num_classes,
                   const int preNmsTop,
                   const int nmsMaxOut,
                   const float iouThreshold,
                   const float minBoxSize,
                   uint8_t* workspace,
                   {{prefix}}Stream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{num_detections}},
{{indent}}   {{detection_boxes}},
{{indent}}   {{detection_scores}},
{{indent}}   {{detection_classe}},
{{indent}}    {{proposals}},
{{indent}}    {{fgScores}},
{{indent}}    {{p_batch}},
{{indent}}    {{num_rois}},
{{indent}}    {{num_classes}},
{{indent}}    {{preNmsTop}},
{{indent}}    {{nmsMaxOut}},
{{indent}}    {{iouThreshold}},
{{indent}}    {{minBoxSize}},
{{indent}}    global_workspace_, stream /* default stream */
{{indent}});
    """
)


def gen_function(func_attrs: Dict[str, Any], header_files, backend_spec) -> str:
    """the function for generating nms kernel"""
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    return FUNC_TEMPLATE.render(
        header_files=header_files,
        kernel=kernel.render(
            prefix=backend_spec.prefix,
            cub=backend_spec.cub,
            elem_input_type=elem_input_type,
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=backend_spec.prefix
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec):
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=backend_spec.prefix
        ).strip()
    )


def gen_function_call(func_attrs, backend_spec, indent="  "):
    """the function for generating a function call for nms op"""

    assert len(func_attrs["outputs"]) == 4
    assert len(func_attrs["inputs"]) == 2

    num_detections = func_attrs["outputs"][0]._attrs["name"]
    detection_boxes = func_attrs["outputs"][1]._attrs["name"]
    detection_scores = func_attrs["outputs"][2]._attrs["name"]
    detection_classes = func_attrs["outputs"][3]._attrs["name"]
    (input_name, score_name) = (
        input_tensor._attrs["name"] for input_tensor in func_attrs["inputs"]
    )

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        num_detections=num_detections,
        detection_boxes=detection_boxes,
        detection_scores=detection_scores,
        detection_classe=detection_classes,
        proposals=input_name,
        fgScores=score_name,
        p_batch="&" + xshape[0]._attrs["name"],
        num_rois="&" + xshape[1]._attrs["name"],
        num_classes="&" + xshape[2]._attrs["name"],
        preNmsTop=func_attrs["preNmsTop"],
        nmsMaxOut=func_attrs["nmsMaxOut"],
        iouThreshold=func_attrs["iouThreshold"],
        minBoxSize=func_attrs["minBoxSize"],
        indent=indent,
    )


def add_profiler(file_pairs, workdir, op_type, output_name, code):
    """generate nms kernel for profiling"""
    prefix = os.path.join(workdir, "profiler", op_type)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    src_path = os.path.join(prefix, output_name + ".cu")
    obj_path = os.path.join(prefix, output_name)
    if os.path.exists(obj_path):
        return
    with open(src_path, "w") as f:
        f.write(code)
    file_pairs.append((src_path, obj_path))


def gen_profiler(func_attrs, workdir, header_files, backend_spec):
    """the function for generating profiler for nms op"""
    op_type = func_attrs["op"]
    file_pairs = []
    elem_input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    code = PROFILER_TEMPLATE.render(
        header_files=header_files,
        elem_input_type=elem_input_type,
        kernel=kernel.render(
            prefix=backend_spec.prefix,
            cub=backend_spec.cub,
            elem_input_type=elem_input_type,
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=backend_spec.prefix
        ),
    )
    op_name = func_attrs["op"]
    add_profiler(file_pairs, workdir, op_type, op_name, code)
    return file_pairs
