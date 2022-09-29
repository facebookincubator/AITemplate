aitemplate.compiler.transform
==============================


apply_padding
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.apply_padding
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:


bind_constants
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.bind_constants
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

constant_folding
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.constant_folding
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor
   :autosummary:

fuse_conv_elementwise
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.fuse_conv_elementwise
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor
   :autosummary:

fuse_group_ops
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.fuse_group_ops
   :members: 
   :imported-members: 
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor, all_static_dimensions
   :autosummary:


fuse_mm_elementwise
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.fuse_mm_elementwise
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor, FuncEnum, elementwise, gemm_rcr, gemm_rcr_bias, gemm_rcr_bias_swish, copy_tensor_attributes, extract_only_one_op, get_patterns, remove_single_tensor_op_from_sorted_graph, sanitize_sorted_graph
   :autosummary:

fuse_ops
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.fuse_ops
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor, FuncEnum, elementwise, gemm_rcr, gemm_rcr_bias, gemm_rcr_bias_swish, copy_tensor_attributes, extract_only_one_op, get_patterns, remove_single_tensor_op_from_sorted_graph, sanitize_sorted_graph, layernorm_sigmoid_mul
   :autosummary:

fuse_parallel_gemms
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.fuse_parallel_gemms
   :members:
   :imported-members:
   :exclude-members: TensorAccessor, is_static_dimension, DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor, FuncEnum, elementwise, gemm_rcr, gemm_rcr_bias, gemm_rcr_bias_swish, copy_tensor_attributes, extract_only_one_op, get_patterns, remove_single_tensor_op_from_sorted_graph, sanitize_sorted_graph, layernorm_sigmoid_mul
   :autosummary:

fuse_permute_bmm
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.fuse_permute_bmm
   :members:
   :imported-members:
   :exclude-members: copy_src_op_attributes, remove_tensor_from_sorted_graph, bmm_ccr, bmm_crr, bmm_rcr, bmm_rrr, gemm_rrr, gemm_rrr_bias, permute021, TensorAccessor, is_static_dimension, DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor, FuncEnum, elementwise, gemm_rcr, gemm_rcr_bias, gemm_rcr_bias_swish, copy_tensor_attributes, extract_only_one_op, get_patterns, remove_single_tensor_op_from_sorted_graph, sanitize_sorted_graph, layernorm_sigmoid_mul
   :autosummary:

fuse_split
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.fuse_split
   :members:
   :imported-members: _fuse_split_and_strided_op
   :exclude-members: IntVar, copy_src_op_attributes, remove_tensor_from_sorted_graph, bmm_ccr, bmm_crr, bmm_rcr, bmm_rrr, gemm_rrr, gemm_rrr_bias, permute021, TensorAccessor, is_static_dimension, DimInfo, IntImm, Operator, Source, Tensor, gemm, AITData, replace_tensor, FuncEnum, elementwise, gemm_rcr, gemm_rcr_bias, gemm_rcr_bias_swish, copy_tensor_attributes, extract_only_one_op, get_patterns, remove_single_tensor_op_from_sorted_graph, sanitize_sorted_graph, layernorm_sigmoid_mul
   :autosummary:

mark_param_tensor
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.mark_param_tensor
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

memory_planning
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.memory_planning
   :members:
   :imported-members:
   :exclude-members: TensorUsageRecord, Workspace, assign_offsets_to_views_and_outputs, greedy_by_size_memory_planning, DimInfo, IntImm, Operator, Source, Tensor, gemm, defaultdict, dataclass
   :autosummary:

name_graph
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.name_graph
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:


optimize_graph
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.optimize_graph
   :members:
   :imported-members:
   :exclude-members: transform_strided_ops, transform_special_ops, transform_odd_alignment, transform_memory_ops, fuse_permute_bmm, fuse_parallel_gemms, fuse_mm_elementwise, apply_padding, fuse_conv_elementwise, fuse_group_ops, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:


profile
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.profile
   :members:
   :imported-members:
   :exclude-members: DynamicProfileStrategy, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

refine_graph
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.refine_graph
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

remove_no_ops
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.remove_no_ops
   :members:
   :imported-members:
   :exclude-members: IntVar, is_singleton_dimension, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

remove_unused_ops
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.remove_unused_ops
   :members:
   :imported-members:
   :exclude-members: deque, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

toposort
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.toposort
   :members:
   :imported-members:
   :exclude-members: DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:


transform_memory_ops
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.transform_memory_ops
   :members:
   :imported-members:
   :exclude-members: TensorAccessor, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

transform_odd_alignment
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.transform_odd_alignment
   :members:
   :imported-members:
   :exclude-members: can_be_constant_folded, copy_src_op_attributes, copy_tensor_attributes, extract_only_one_op, remove_tensor_from_sorted_graph, replace_tensor, sanitize_sorted_graph, toposort, IntVar, bmm_ccr, bmm_crr, bmm_rcr, bmm_rrr, permute021, unsqueeze, TensorAccessor, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:


transform_special_ops
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.transform_special_ops
   :members:
   :imported-members:
   :exclude-members: gemm_rrr, is_singleton_dimension, gemm_rcr, gemm_rrr_small_nk, can_be_constant_folded, copy_src_op_attributes, copy_tensor_attributes, extract_only_one_op, remove_tensor_from_sorted_graph, replace_tensor, sanitize_sorted_graph, toposort, IntVar, bmm_ccr, bmm_crr, bmm_rcr, bmm_rrr, permute021, unsqueeze, TensorAccessor, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

transform_strided_op_and_view_op
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.transform_strided_op_and_view_op
   :members:
   :imported-members:
   :exclude-members: IntVar, is_singleton_dimension, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

transform_strided_ops
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.transform_strided_ops
   :members:
   :imported-members:
   :exclude-members: get_tensor_index, slice_reshape_scatter, slice_scatter, gen_tensor_index, IntVar, is_singleton_dimension, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:

transform_strided_slice
-------------------------------------------
.. automodule:: aitemplate.compiler.transform.transform_strided_slice
   :members:
   :imported-members:
   :exclude-members: dynamic_slice, get_tensor_index, slice_reshape_scatter, slice_scatter, gen_tensor_index, IntVar, is_singleton_dimension, DimInfo, IntImm, Operator, Source, Tensor, gemm
   :autosummary:
