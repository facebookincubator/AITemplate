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
from .acc_ops import *  # isort:skip # noqa: F403 F401
from .ait_acc_ops import *  # noqa: F403 F401
import logging

from .acc_normalizer import (
    _normalization_dict,
    register_acc_op_mapping,
    register_custom_acc_mapper_fn,
)

from .ait_acc_ops_registry import get_ait_acc_op_mappers, get_custom_ait_acc_op_mappers

logger: logging.Logger = logging.getLogger(__name__)


def update_acc_op_mappers_for_ait() -> None:
    """
    This function allows to replace some of the default acc_ops mappers with
    custom mappers. Custom mappers are defined in the 'ait_acc_ops.py' file.
    """
    ait_acc_op_mappers = get_ait_acc_op_mappers()
    custom_ait_acc_op_mappers = get_custom_ait_acc_op_mappers()

    logger.info(
        "Found %s ait mappers, %s custom ait op mappers",
        len(ait_acc_op_mappers),
        len(custom_ait_acc_op_mappers),
    )

    for op_and_target, mapper in ait_acc_op_mappers.items():
        if op_and_target in _normalization_dict:
            logger.info("Removing %s from acc normalization dict", op_and_target)
            del _normalization_dict[op_and_target]

        logger.info("Adding AIT acc mapper for %s", op_and_target)
        register_acc_op_mapping(
            op_and_target,
            mapper.arg_replacement_tuples,
            mapper.kwargs_to_move_to_acc_out_ty,
        )(mapper.new_fn_target)

    for op_and_target, mapper in custom_ait_acc_op_mappers.items():
        if op_and_target in _normalization_dict:
            logger.info("Removing %s from acc normalization dict", op_and_target)
            del _normalization_dict[op_and_target]

        logger.info("Adding custom AIT acc mapper for %s", op_and_target)
        register_custom_acc_mapper_fn(
            op_and_target,
            mapper.arg_replacement_tuples,
            mapper.needs_shapes_for_normalization,
            mapper.allow_normalize_from_torch_package,
        )(mapper.custom_mapping_fn)

    logger.info("Completed updating acc mappers")
