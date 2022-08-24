# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] cuda reduce common functions
"""
import jinja2

from ....compiler.base import IntImm, IntVar
from .. import cuda_common

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  {{elem_output_type}}* /*dst_ptr*/,
  {{elem_input_type}}*  /*src_ptr*/,
  int                   /*reduction_axis*/,
  const int64_t*        /*shape*/,
  const int             /*rank*/,
  cudaStream_t
);
"""
)


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if (shape[rank - 1] % {{vector_length}} == 0) {
{{indent}}  {{func_name}}_launcher<{{vector_length}}>(
{{indent}}      dst_ptr, src_ptr, reduction_axis, shape, rank, stream);
{{indent}}  return;
}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <cassert>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/reduction/device/tensor_reduce.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/util/host_tensor.h"
#ifndef CUTLASS_CHECK_REDUCE
#define CUTLASS_CHECK_REDUCE(status)                                                             \\
  {                                                                                              \\
    cutlass::Status error = status;                                                              \\
    if (error != cutlass::Status::kSuccess) {                                                    \\
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \\
                << std::endl;                                                                    \\
      exit(EXIT_FAILURE);                                                                        \\
    }                                                                                            \\
  }
#endif
template <int VectorLength = 1>
void {{func_name}}_launcher(
    {{elem_output_type}} *dst_ptr,
    {{elem_input_type}} *src_ptr,
    int reduction_axis,
    const int64_t *shape,
    const int rank,
    cudaStream_t stream) {
  // Instead of making our own 4D tensor definition,
  // we simply use TensoeNHWC as a 4D tensor
  using Layout = cutlass::layout::TensorNHWC;
  using ElementCompute = {{elem_compute_type}};
  using ReductionOp = {{reduction_op}}<ElementCompute>;
  constexpr int NUM_DIMS = 4;
  assert(rank <= NUM_DIMS);
  assert(reduction_axis < rank);
  assert(rank > 0);
  using TensorReduction = cutlass::reduction::device::TensorReduction<
    {{elem_output_type}},
    {{elem_input_type}},
    Layout,
    ReductionOp,
    VectorLength,
    ElementCompute
  >;
  assert(shape[rank - 1] % VectorLength == 0);
  // adjust reduction_axis
  reduction_axis = NUM_DIMS - rank + reduction_axis;
  // cutlass's tensor_reduce only supports 4D tensors at the moment
  int64_t dst_dims[NUM_DIMS];
  int64_t src_dims[NUM_DIMS];
  for (int i = 0; i < NUM_DIMS; i++) {
    dst_dims[i] = 1;
    src_dims[i] = 1;
  }
  for (int i = 0; i < rank; i++) {
    int idx = NUM_DIMS - rank + i;
    dst_dims[idx] = shape[i];
    src_dims[idx] = shape[i];
  }
  dst_dims[reduction_axis] = 1;
  Layout::TensorCoord dst_extent(
    dst_dims[0], dst_dims[1], dst_dims[2], dst_dims[3]
  );
  Layout dst_layout(Layout::packed(dst_extent));
  Layout::TensorCoord src_extent(
    src_dims[0], src_dims[1], src_dims[2], src_dims[3]
  );
  Layout src_layout(Layout::packed(src_extent));
  ElementCompute reduction_identity = ElementCompute();
  TensorReduction reduction(src_extent, reduction_axis);
  cutlass::DeviceAllocation<uint8_t> workspace(reduction.workspace_size());
  ReductionOp reduction_op = ReductionOp();
  assert(dst_ptr);
  assert(src_ptr);
  cutlass::Status status = reduction.reduce(
      {dst_ptr, dst_layout},
      {src_ptr, src_layout},
      workspace.get(),
      reduction_identity,
      reduction_op,
      stream
    );
  CUTLASS_CHECK_REDUCE(status);
}
#undef CUTLASS_CHECK_REDUCE
void {{func_name}}(
    {{elem_output_type}} *dst_ptr,
    {{elem_input_type}}  *src_ptr,
    int reduction_axis,
    const int64_t *shape,
    const int rank,
    cudaStream_t stream) {
  if (!dst_ptr) {
    throw std::runtime_error("dst_ptr is nullptr!");
  }
  if (!src_ptr) {
    throw std::runtime_error("src_ptr is nullptr!");
  }
  {{exec_paths}}
  throw std::runtime_error(
    "Unsupported workload for this {{func_name}} specialization."
  );
}
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
  {{indent}}int64_t shape[] = {{dims}};
  {{indent}}{{func_name}}(
  {{indent}}    {{dst_ptr}},
  {{indent}}    {{src_ptr}},
  {{indent}}    {{reduction_axis}},
  {{indent}}    shape,
  {{indent}}    {{rank}},
  {{indent}}    stream
  {{indent}});
{{indent}}}
"""
)


def gen_function_decl(func_attrs):
    """[summary]
    Parameters
    ----------
    func_attrs : Dit[str, Any]
        [description]
    Returns
    -------
    [type] : str
        [description]
    """
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        elem_input_type=cuda_common.dtype_to_cutlass_type(x._attrs["dtype"]),
        elem_output_type=cuda_common.dtype_to_cutlass_type(y._attrs["dtype"]),
    )


def gen_function(func_attrs, reduction_op):
    """[summary]
    Parameters
    ----------
    func_attrs : Dict[str, Any]
        [description]
    Returns
    -------
    [type] : str
        [description]
    """
    vector_lens_config = [32, 16, 8, 4, 1]
    exec_paths = ""
    for vlen in vector_lens_config:
        exec_program = EXEC_COND_TEMPLATE.render(
            func_name=func_attrs["name"], vector_length=vlen, indent="  "
        )
        exec_paths += exec_program

    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    input_type = cuda_common.dtype_to_cutlass_type(x._attrs["dtype"])
    output_type = cuda_common.dtype_to_cutlass_type(y._attrs["dtype"])
    return SRC_TEMPLATE.render(
        func_name=func_attrs["name"],
        elem_input_type=input_type,
        elem_output_type=output_type,
        # Match pytorch's behavior where the accumuation type is the same
        # as the output type
        elem_compute_type=output_type,
        reduction_op=reduction_op,
        exec_paths=exec_paths,
    )


def gen_function_call(func_attrs, indent="  "):
    """[summary]
    Parameters
    ----------
    func_attrs : Dict[str, Any]
        [description]
    indent : str, optional
        [description], by default "  "
    Returns
    -------
    [type] str
        [description]
    """
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    axes = func_attrs["reduction_axes"]
    if not len(axes) == 1:
        raise NotImplementedError("Multiple reduction axes are not supported yet")

    def dim_to_str(dim):
        if isinstance(dim, IntVar):
            return dim._attrs["name"]
        if isinstance(dim, IntImm):
            return str(dim._attrs["values"][0])
        raise NotImplementedError("Unsupported dim kind: {dim}".format(dim=dim))

    x_shape = x._attrs["shape"]
    dims = ",".join([dim_to_str(dim) for dim in x_shape])

    return FUNC_CALL_TEMPLATE.render(
        indent=indent,
        func_name=func_attrs["name"],
        dst_ptr=y._attrs["name"],
        src_ptr=x._attrs["name"],
        reduction_axis=axes[0],
        dims="{ " + dims + " }",
        rank=str(len(x_shape)),
    )
