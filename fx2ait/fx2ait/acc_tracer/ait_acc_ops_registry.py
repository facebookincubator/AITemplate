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
from collections.abc import Callable
from typing import NamedTuple


class AitAccOpMapper(NamedTuple):
    new_fn_target: Callable
    arg_replacement_tuples: (
        None
        | (
            list[
                (
                    tuple[str | tuple[str, ...], str]
                    | tuple[str | tuple[str, ...], str, bool]
                )
            ]
        )
    )
    kwargs_to_move_to_acc_out_ty: None | (list[tuple[str, str, bool] | tuple[str, str]])


class CustomAitAccOpMapper(NamedTuple):
    custom_mapping_fn: Callable
    arg_replacement_tuples: list[
        (tuple[str | tuple[str, ...], str] | tuple[str | tuple[str, ...], str, bool])
    ]
    needs_shapes_for_normalization: bool
    allow_normalize_from_torch_package: bool


_AIT_ACC_OP_MAPPERS: dict[tuple[str, str | Callable], AitAccOpMapper] = {}
_CUSTOM_AIT_ACC_OP_MAPPERS: dict[tuple[str, str | Callable], CustomAitAccOpMapper] = {}


def ait_register_acc_op_mapping(
    op_and_target: tuple[str, str | Callable],
    arg_replacement_tuples: None
    | (
        list[
            (
                tuple[str | tuple[str, ...], str]
                | tuple[str | tuple[str, ...], str, bool]
            )
        ]
    ) = None,
    kwargs_to_move_to_acc_out_ty: None
    | (list[tuple[str, str, bool] | tuple[str, str]]) = None,
):
    def insert(new_fn_target: Callable):
        _AIT_ACC_OP_MAPPERS[op_and_target] = AitAccOpMapper(
            new_fn_target=new_fn_target,
            arg_replacement_tuples=arg_replacement_tuples,
            kwargs_to_move_to_acc_out_ty=kwargs_to_move_to_acc_out_ty,
        )
        return new_fn_target

    return insert


def ait_register_custom_acc_mapper_fn(
    op_and_target: tuple[str, str | Callable],
    arg_replacement_tuples: list[
        (tuple[str | tuple[str, ...], str] | tuple[str | tuple[str, ...], str, bool])
    ],
    needs_shapes_for_normalization=False,
    allow_normalize_from_torch_package=False,
):
    def insert(custom_mapping_fn: Callable):
        _CUSTOM_AIT_ACC_OP_MAPPERS[op_and_target] = CustomAitAccOpMapper(
            custom_mapping_fn=custom_mapping_fn,
            arg_replacement_tuples=arg_replacement_tuples,
            needs_shapes_for_normalization=needs_shapes_for_normalization,
            allow_normalize_from_torch_package=allow_normalize_from_torch_package,
        )
        return custom_mapping_fn

    return insert


def get_ait_acc_op_mappers() -> dict[tuple[str, str | Callable], AitAccOpMapper]:
    return _AIT_ACC_OP_MAPPERS


def get_custom_ait_acc_op_mappers() -> (
    dict[tuple[str, str | Callable], CustomAitAccOpMapper]
):
    return _CUSTOM_AIT_ACC_OP_MAPPERS
