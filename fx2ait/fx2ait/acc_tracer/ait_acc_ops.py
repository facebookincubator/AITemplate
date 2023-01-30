import torch

from fx2ait.acc_tracer.acc_normalizer import register_acc_op

from fx2ait.acc_tracer.ait_acc_ops_registry import ait_register_acc_op_mapping


@ait_register_acc_op_mapping(
    op_and_target=("call_method", "split"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@ait_register_acc_op_mapping(
    op_and_target=("call_function", torch.split),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_acc_op
def split(*, input, split_size_or_sections, dim=0):
    return torch.split(input, split_size_or_sections, dim)
