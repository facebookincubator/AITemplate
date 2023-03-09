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
nms kernel codegen.
"""

import os
from typing import Any, Dict, List

import jinja2

from aitemplate.backend.common.vision_ops.nms_kernel import KERNEL_TEMPLATE

# pylint: disable=C0301

FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

const int T_SIZE = {{T_SIZE}}; //(preNmsTopN + blockSize - 1) / blockSize - 1;
{{kernel}}

}  // namespace

{{func_signature}}
{

    const int N = *batch;
    const int R = *num_rois;
    nmsGpu<{{elem_scores_type}}, {{elem_rois_type}}>(
        stream, N, R, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize,
        fgScores, proposals, workspace, rois);
}
    """
)

PROFILER_TEMPLATE = jinja2.Template(
    """
#include <iostream>
{{header_files}}


size_t GLOBAL_WORKSPACE_SIZE = 0;

namespace {

const int T_SIZE = {{T_SIZE}}; //(preNmsTopN + blockSize - 1) / blockSize - 1;
{{kernel}}

}  // namespace

int main(int argc, char** argv) {
  int instance_num = std::stoi(argv[1]); // batch
  int instance_size = std::stoi(argv[2]); // num_rois
  int elem_cnt = instance_size * instance_num;

  float runtime_ms = 0;
  const int64_t offsets_bytes = GetCudaAlignedSize((instance_num+1) * sizeof(int64_t));
  const int64_t scores_bytes = GetCudaAlignedSize(elem_cnt * sizeof({{elem_scores_type}}));
  const int64_t boxes_bytes = GetCudaAlignedSize(elem_cnt * 4 * sizeof({{elem_rois_type}}));
  int64_t temp_storage_bytes = InferTempStorageForSortPairsDescending<{{elem_scores_type}}, int64_t>(instance_num, instance_size);

  GLOBAL_WORKSPACE_SIZE = GetCudaAlignedSize(offsets_bytes + scores_bytes + boxes_bytes + temp_storage_bytes);

  std::cout << "TIME:" << runtime_ms << std::endl;
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* rois,
                   const void* proposals,
                   const void* fgScores,
                   int64_t* batch,
                   int64_t* num_rois,
                   const {{index_type}} preNmsTop,
                   const {{index_type}} nmsMaxOut,
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
{{indent}}   {{rois}}, {{proposals}}, {{fgScores}},
{{indent}}    {{p_batch}},
{{indent}}    {{num_rois}},
{{indent}}    {{preNmsTop}},
{{indent}}    {{nmsMaxOut}},
{{indent}}    {{iouThreshold}},
{{indent}}    {{minBoxSize}},
{{indent}}    global_workspace_, stream /* default stream */
{{indent}});
    """
)


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    """the function for generating nms kernel"""
    blockSize = 1024
    t_size = int((func_attrs["preNmsTop"] + blockSize - 1) / blockSize)
    if backend_spec.backend_name == "cuda":
        cuda_hmaxmin = True
    else:
        cuda_hmaxmin = False

    elem_rois_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_scores_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][1]._attrs["dtype"]
    )
    return FUNC_TEMPLATE.render(
        T_SIZE=t_size,
        elem_scores_type=elem_scores_type,
        elem_rois_type=elem_rois_type,
        header_files=header_files,
        kernel=KERNEL_TEMPLATE.render(
            prefix=backend_spec.prefix, cub=backend_spec.cub, cuda_hmaxmin=cuda_hmaxmin
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
            index_type=backend_spec.index_type,
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
            index_type=backend_spec.index_type,
        ).strip()
    )


def gen_function_call(func_attrs: Dict[str, Any], backend_spec, indent: str) -> str:
    """ "The function for generating a function call to nms"""
    output_name = ""
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 2

    output_name = func_attrs["outputs"][0]._attrs["name"]
    (input_name, score_name) = (
        input_tensor._attrs["name"] for input_tensor in func_attrs["inputs"]
    )

    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        rois=output_name,
        proposals=input_name,
        fgScores=score_name,
        p_batch="&" + xshape[0]._attrs["name"],
        num_rois="&" + xshape[1]._attrs["name"],
        preNmsTop=func_attrs["preNmsTop"],
        nmsMaxOut=func_attrs["nmsMaxOut"],
        iouThreshold=func_attrs["iouThreshold"],
        minBoxSize=func_attrs["minBoxSize"],
        indent=indent,
    )


def add_profiler(
    file_pairs: List[Any], workdir: str, op_type, output_name: str, code: str
) -> None:
    """generate code for profiling"""
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


def gen_profiler(
    func_attrs: Dict[str, Any], workdir: str, header_files: str, backend_spec
) -> None:
    """generate and build code for NMS profiling"""
    op_type = func_attrs["op"]
    file_pairs = []
    blockSize = 1024
    t_size = int((func_attrs["preNmsTop"] + blockSize - 1) / blockSize)

    if backend_spec.backend_name == "cuda":
        cuda_hmaxmin = True
    else:
        cuda_hmaxmin = False

    elem_rois_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_scores_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][1]._attrs["dtype"]
    )
    code = PROFILER_TEMPLATE.render(
        T_SIZE=t_size,
        elem_scores_type=elem_scores_type,
        elem_rois_type=elem_rois_type,
        header_files=header_files,
        kernel=KERNEL_TEMPLATE.render(
            prefix=backend_spec.prefix, cub=backend_spec.cub, cuda_hmaxmin=cuda_hmaxmin
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
            index_type=backend_spec.index_type,
        ),
    )
    op_name = func_attrs["op"]
    add_profiler(file_pairs, workdir, op_type, op_name, code)
    return file_pairs
