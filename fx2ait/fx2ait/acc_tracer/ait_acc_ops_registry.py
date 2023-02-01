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
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union


class AitAccOpMapper(NamedTuple):
    new_fn_target: Callable
    arg_replacement_tuples: Optional[
        List[
            Union[
                Tuple[Union[str, Tuple[str, ...]], str],
                Tuple[Union[str, Tuple[str, ...]], str, bool],
            ]
        ]
    ]
    kwargs_to_move_to_acc_out_ty: Optional[
        List[Union[Tuple[str, str, bool], Tuple[str, str]]]
    ]


class CustomAitAccOpMapper(NamedTuple):
    custom_mapping_fn: Callable
    arg_replacement_tuples: List[
        Union[
            Tuple[Union[str, Tuple[str, ...]], str],
            Tuple[Union[str, Tuple[str, ...]], str, bool],
        ]
    ]
    needs_shapes_for_normalization: bool
    allow_normalize_from_torch_package: bool


_AIT_ACC_OP_MAPPERS: Dict[Tuple[str, Union[str, Callable]], AitAccOpMapper] = {}
_CUSTOM_AIT_ACC_OP_MAPPERS: Dict[
    Tuple[str, Union[str, Callable]], CustomAitAccOpMapper
] = {}


def ait_register_acc_op_mapping(
    op_and_target: Tuple[str, Union[str, Callable]],
    arg_replacement_tuples: Optional[
        List[
            Union[
                Tuple[Union[str, Tuple[str, ...]], str],
                Tuple[Union[str, Tuple[str, ...]], str, bool],
            ]
        ]
    ] = None,
    kwargs_to_move_to_acc_out_ty: Optional[
        List[Union[Tuple[str, str, bool], Tuple[str, str]]]
    ] = None,
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
    op_and_target: Tuple[str, Union[str, Callable]],
    arg_replacement_tuples: List[
        Union[
            Tuple[Union[str, Tuple[str, ...]], str],
            Tuple[Union[str, Tuple[str, ...]], str, bool],
        ]
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


def get_ait_acc_op_mappers() -> Dict[Tuple[str, Union[str, Callable]], AitAccOpMapper]:
    return _AIT_ACC_OP_MAPPERS


def get_custom_ait_acc_op_mappers() -> (
    Dict[Tuple[str, Union[str, Callable]], CustomAitAccOpMapper]
):
    return _CUSTOM_AIT_ACC_OP_MAPPERS
