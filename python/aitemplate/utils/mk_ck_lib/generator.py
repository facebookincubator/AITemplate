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

import copy

from aitemplate.utils.mk_ck_lib import (
    conv2d_operation as conv,
    gemm_operation as gemm,
    groupnorm_operation as groupnorm,
    layernorm_operation as layernorm,
    library,
    softmax_operation as softmax,
)


###########################################################################################################
# Convolution for 2D Fwd operations
def CreateConv2dFwdOperator(manifest, operation_kind, out_element_op, out_data_op=""):
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.G_NHW_C
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.G_K_YX_C
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.G_NHW_K
    )

    in_element_op = library.TensorOperation.PassThrough

    tile_descriptions = [
        conv.GroupTileDesc(1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        conv.GroupTileDesc(1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        conv.GroupTileDesc(1, 128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        conv.GroupTileDesc(1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        conv.GroupTileDesc(1, 128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        conv.GroupTileDesc(1, 128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        conv.GroupTileDesc(1, 64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        conv.GroupTileDesc(1, 256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        conv.GroupTileDesc(1, 256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        conv.GroupTileDesc(1, 128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        conv.GroupTileDesc(1, 64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        conv.GroupTileDesc(1, 64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    c_block_descriptions = [
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8),
        conv.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8),
        conv.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8),
    ]

    block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
        assert (
            block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            conv.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )

    conv2d_specialization = [
        conv.Conv2DSpecialization.ConvFwdDefault,
        conv.Conv2DSpecialization.ConvFwd1x1P0,
        conv.Conv2DSpecialization.ConvFwd1x1S1P0,
    ]

    gemm_specialization = [
        conv.Conv2DSpecialization.GemmDefault,
        conv.Conv2DSpecialization.MNKPadding,
    ]

    operations = []
    for conv2d_spec in conv2d_specialization:
        for gemm_spec in gemm_specialization:
            for tile_desc, block_desc, c_block_desc in zip(
                tile_descriptions, block_descriptions, c_block_descriptions
            ):
                new_operation = conv.Conv2DOperation(
                    operation_kind=operation_kind,
                    extra_kind=out_element_op,
                    xdl_op_type=conv.XdlOpType(operation_kind.value),
                    A=a_element_desc,
                    B=b_element_desc,
                    C=c_element_desc,
                    a_elem_op=in_element_op,
                    b_elem_op=in_element_op,
                    epilogue_functor=out_element_op,
                    c_data_op=out_data_op,
                    conv2d_specialization=conv2d_spec,
                    gemm_specialization=gemm_spec,
                    tile_desc=tile_desc,
                    a_block_transfer=block_desc,
                    b_block_transfer=block_desc,
                    c_block_transfer=c_block_desc,
                )
                manifest.append(new_operation)
                operations.append(new_operation)

    conv2d_specialization = [conv.Conv2DSpecialization.ConvFwdOddC]

    tile_descriptions += [
        conv.GroupTileDesc(1, 256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        conv.GroupTileDesc(1, 256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        conv.GroupTileDesc(1, 256, 256, 64, 32, 8, 8, 32, 32, 4, 1),
        conv.GroupTileDesc(1, 128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        conv.GroupTileDesc(1, 128, 64, 64, 32, 8, 8, 32, 32, 1, 2),
        conv.GroupTileDesc(1, 256, 256, 16, 32, 8, 8, 16, 16, 4, 1),  # c_out=1
    ]

    block_descriptions = [
        conv.BlockTransferDesc([4, 8, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 8, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 4, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 8, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 4, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 4, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 2, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 8, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 8, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 4, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 2, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 2, 8], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([2, 32, 4], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([2, 32, 4], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([2, 32, 4], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([2, 16, 4], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([2, 16, 4], [1, 0, 2], [1, 0, 2], 2, 1, 1, 1),
        conv.BlockTransferDesc([4, 16, 4], [1, 0, 2], [1, 0, 2], 2, 2, 2, 1),  # c_out=1
    ]

    c_block_descriptions += [
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        conv.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8),
        conv.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8),
        conv.CBlockTransferDesc(4, 1, [1, 256, 1, 1], 1),  # c_out=1
    ]
    for conv2d_spec in conv2d_specialization:
        for gemm_spec in gemm_specialization:
            for tile_desc, block_desc, c_block_desc in zip(
                tile_descriptions, block_descriptions, c_block_descriptions
            ):
                new_operation = conv.Conv2DOperation(
                    operation_kind=operation_kind,
                    extra_kind=out_element_op,
                    xdl_op_type=conv.XdlOpType(operation_kind.value),
                    A=a_element_desc,
                    B=b_element_desc,
                    C=c_element_desc,
                    a_elem_op=in_element_op,
                    b_elem_op=in_element_op,
                    epilogue_functor=out_element_op,
                    c_data_op=out_data_op,
                    conv2d_specialization=conv2d_spec,
                    gemm_specialization=gemm_spec,
                    tile_desc=tile_desc,
                    a_block_transfer=block_desc,
                    b_block_transfer=block_desc,
                    c_block_transfer=c_block_desc,
                )
                manifest.append(new_operation)
                operations.append(new_operation)
    return operations


###########################################################################################################
# Gemm operations
def CreateGemmRRROperator(manifest):
    operation_kind = library.GemmKind.Gemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough

    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 2, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 2, 32, 32, 2, 1),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 2, 32, 32, 1, 2),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
    ]

    b_block_descriptions = [
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([8, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([16, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
    ]
    a_block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        a_block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            a_block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            a_block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            if t.n_per_block == 64:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)

        assert (
            a_block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        a_block_descriptions.append(
            gemm.BlockTransferDesc(a_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)

    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    
    loop_schedulers = ["ck::LoopScheduler::Default", "ck::LoopScheduler::Interwave"]
    pipelines = ["ck::PipelineVersion::v1", "ck::PipelineVersion::v2"]

    operations = []
    for gemm_spec in gemm_specialization:
        for loop_scheduler in loop_schedulers:
            for pipeline in pipelines:
                if pipeline == "ck::PipelineVersion::v2" and loop_scheduler == "ck::LoopScheduler::Interwave":
                    continue
                for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
                    tile_descriptions,
                    a_block_descriptions,
                    b_block_descriptions,
                    c_block_descriptions,
                ):
                    new_operation = gemm.GemmOperation(
                        operation_kind=operation_kind,
                        extra_kind=element_op,
                        xdl_op_type=gemm.XdlOpType.DeviceGemmXdl_CShuffle,
                        A=a_element_desc,
                        B=b_element_desc,
                        C=c_element_desc,
                        a_elem_op=element_op,
                        b_elem_op=element_op,
                        epilogue_functor=element_op,
                        gemm_specialization=gemm_spec,
                        tile_desc=tile_desc,
                        a_block_transfer=a_block_desc,
                        b_block_transfer=b_block_desc,
                        c_block_transfer=c_block_desc,
                        loop_scheduler=loop_scheduler,
                        pipeline=pipeline,
                    )
                    manifest.append(new_operation)
                    operations.append(new_operation)
    return operations


def CreateGemmRCROperator(manifest):
    operation_kind = library.GemmKind.Gemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough

    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8)

        assert (
            block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]

    loop_schedulers = ["ck::LoopScheduler::Default", "ck::LoopScheduler::Interwave"]
    pipelines = ["ck::PipelineVersion::v1", "ck::PipelineVersion::v2"]

    operations = []
    for gemm_spec in gemm_specialization:
        for loop_scheduler in loop_schedulers:
            for pipeline in pipelines:
                if pipeline == "ck::PipelineVersion::v2" and loop_scheduler == "ck::LoopScheduler::Interwave":
                    continue
                for tile_desc, block_desc, c_block_desc in zip(
                    tile_descriptions, block_descriptions, c_block_descriptions
                ):
                    new_operation = gemm.GemmOperation(
                        operation_kind=operation_kind,
                        extra_kind=element_op,
                        xdl_op_type=gemm.XdlOpType.DeviceGemmXdl_CShuffle,
                        A=a_element_desc,
                        B=b_element_desc,
                        C=c_element_desc,
                        a_elem_op=element_op,
                        b_elem_op=element_op,
                        epilogue_functor=element_op,
                        gemm_specialization=gemm_spec,
                        tile_desc=tile_desc,
                        a_block_transfer=block_desc,
                        b_block_transfer=block_desc,
                        c_block_transfer=c_block_desc,
                        loop_scheduler=loop_scheduler,
                        pipeline=pipeline
                    )
                    manifest.append(new_operation)
                    operations.append(new_operation)
    return operations


def CreateGemmRCRBilinearOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.Gemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    if c_element_op in [
        library.TensorOperation.AddMulAdd,
        library.TensorOperation.AddAddAdd,
        library.TensorOperation.AddAddAddRelu,
    ]:
        ds_dtype = [library.DataType.f16, library.DataType.f16, library.DataType.f16]
        ds_layout = [
            library.LayoutType.RowMajor,
            library.LayoutType.RowMajor,
            library.LayoutType.RowMajor,
        ]
    elif c_element_op in [
        library.TensorOperation.AddSigmoidMul,
        library.TensorOperation.AddSigmoidMulTanh,
        library.TensorOperation.AddAdd,
        library.TensorOperation.AddMul,
        library.TensorOperation.AddMulTanh,
        library.TensorOperation.AddAddRelu,
    ]:
        ds_dtype = [library.DataType.f16, library.DataType.f16]
        ds_layout = [library.LayoutType.RowMajor, library.LayoutType.RowMajor]
    else:
        ds_dtype = [library.DataType.f16]
        ds_layout = [library.LayoutType.RowMajor]
    e_dtype = library.DataType.f16
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8)

        assert (
            block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]

    loop_schedulers = ["ck::LoopScheduler::Default", "ck::LoopScheduler::Interwave"]
    pipelines = ["ck::PipelineVersion::v1", "ck::PipelineVersion::v2"]

    operations = []
    for gemm_spec in gemm_specialization:
        for loop_scheduler in loop_schedulers:
            for pipeline in pipelines:
                if pipeline == "ck::PipelineVersion::v2" and loop_scheduler == "ck::LoopScheduler::Interwave":
                    continue
                for tile_desc, block_desc, c_block_desc in zip(
                    tile_descriptions, block_descriptions, c_block_descriptions
                ):
                    new_operation = gemm.GemmOperation(
                        operation_kind=operation_kind,
                        extra_kind=c_element_op,
                        xdl_op_type=gemm.XdlOpType.DeviceGemmMultipleD_Xdl_CShuffle,
                        A=a_element_desc,
                        B=b_element_desc,
                        C=c_element_desc,
                        a_elem_op=element_op,
                        b_elem_op=element_op,
                        epilogue_functor=c_element_op,
                        gemm_specialization=gemm_spec,
                        tile_desc=tile_desc,
                        a_block_transfer=block_desc,
                        b_block_transfer=block_desc,
                        c_block_transfer=c_block_desc,
                        ds_dtype=ds_dtype,
                        ds_layout=ds_layout,
                        e_dtype=e_dtype,
                        loop_scheduler=loop_scheduler,
                        pipeline=pipeline
                    )
                    manifest.append(new_operation)
                    operations.append(new_operation)

    if c_element_op in [
        library.TensorOperation.Add,  # gemm_rcr_bias
        library.TensorOperation.AddRelu,  # gemm_rcr_bias_relu
    ]:
        # N % 8 == 0 && K % 1 == 0
        gemm_spec = gemm.GemmSpecialization.MNKPadding
        for loop_scheduler in loop_schedulers:
            for pipeline in pipelines:
                if pipeline == "ck::PipelineVersion::v2" and loop_scheduler == "ck::LoopScheduler::Interwave":
                    continue
                for tile_desc, block_desc, c_block_desc in zip(
                    tile_descriptions, block_descriptions, c_block_descriptions
                ):
                    c_block_desc = copy.deepcopy(c_block_desc)
                    c_block_desc.scalar_per_vector = 1
                    c_block_desc.m_n_block_wave_per_xdl[1] //= 8
                    c_block_desc.m_n_block_wave_per_xdl[-1] *= 8
                    new_operation = gemm.GemmOperation(
                        operation_kind=operation_kind,
                        extra_kind=c_element_op,
                        xdl_op_type=gemm.XdlOpType.DeviceGemmMultipleD_Xdl_CShuffle,
                        A=a_element_desc,
                        B=b_element_desc,
                        C=c_element_desc,
                        a_elem_op=element_op,
                        b_elem_op=element_op,
                        epilogue_functor=c_element_op,
                        gemm_specialization=gemm_spec,
                        tile_desc=tile_desc,
                        a_block_transfer=block_desc,
                        b_block_transfer=block_desc,
                        c_block_transfer=c_block_desc,
                        ds_dtype=ds_dtype,
                        ds_layout=ds_layout,
                        e_dtype=e_dtype,
                        loop_scheduler=loop_scheduler,
                        pipeline=pipeline
                    )
                    manifest.append(new_operation)
                    operations.append(new_operation)

        # N % 4 == 0 && K % 4 == 0
        gemm_spec = gemm.GemmSpecialization.MNKPadding
        for loop_scheduler in loop_schedulers:
            for pipeline in pipelines:
                if pipeline == "ck::PipelineVersion::v2" and loop_scheduler == "ck::LoopScheduler::Interwave":
                    continue
                for tile_desc, block_desc, c_block_desc in zip(
                    tile_descriptions, block_descriptions, c_block_descriptions
                ):
                    block_desc.src_scalar_per_vector = 4
                    block_desc.dst_scalar_per_vector = 4
                    c_block_desc = copy.deepcopy(c_block_desc)
                    c_block_desc.scalar_per_vector = 4
                    c_block_desc.m_n_block_wave_per_xdl[1] //= 2
                    c_block_desc.m_n_block_wave_per_xdl[-1] *= 2
                    new_operation = gemm.GemmOperation(
                        operation_kind=operation_kind,
                        extra_kind=c_element_op,
                        xdl_op_type=gemm.XdlOpType.DeviceGemmMultipleD_Xdl_CShuffle,
                        A=a_element_desc,
                        B=b_element_desc,
                        C=c_element_desc,
                        a_elem_op=element_op,
                        b_elem_op=element_op,
                        epilogue_functor=c_element_op,
                        gemm_specialization=gemm_spec,
                        tile_desc=tile_desc,
                        a_block_transfer=block_desc,
                        b_block_transfer=block_desc,
                        c_block_transfer=c_block_desc,
                        ds_dtype=ds_dtype,
                        ds_layout=ds_layout,
                        e_dtype=e_dtype,
                        loop_scheduler=loop_scheduler,
                        pipeline=pipeline
                    )
                    manifest.append(new_operation)
                    operations.append(new_operation)

    return operations


def CreateBmmRCROperator(manifest):
    operation_kind = library.GemmKind.BatchGemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 4, 8, 0, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 4, 8, 0, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 4, 8, 0, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 32, 4, 8, 0, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 4, 8, 0, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 4, 8, 0, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 4, 8, 0, 32, 32, 1, 2),
    ]

    block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
        if t.block_size == 64:
            block_transfer = [4, 16, 1]

        assert (
            block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(
                block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
            )
        )
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, block_desc in zip(tile_descriptions, block_descriptions):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmXdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=block_desc,
                b_block_transfer=block_desc,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateGemmRCRPermOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.GemmPermute
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    ds_dtype = [library.DataType.f16]
    e_dtype = library.DataType.f16
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        # gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
            # TODO:figure out the last dimension
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 1)
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 1)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 1)
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 1)

        assert (
            block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]

    loop_schedulers = ["ck::LoopScheduler::Default", "ck::LoopScheduler::Interwave"]
    pipelines = ["ck::PipelineVersion::v1", "ck::PipelineVersion::v2"]

    operations = []
    for loop_scheduler in loop_schedulers:
        for pipeline in pipelines:
            if pipeline == "ck::PipelineVersion::v2" and loop_scheduler == "ck::LoopScheduler::Interwave":
                continue
            for gemm_spec in gemm_specialization:
                for tile_desc, block_desc, c_block_desc in zip(
                    tile_descriptions, block_descriptions, c_block_descriptions
                ):
                    new_operation = gemm.GemmOperation(
                        operation_kind=operation_kind,
                        extra_kind=c_element_op,
                        xdl_op_type=gemm.XdlOpType.DeviceGemmBiasCPermute_Xdl,
                        A=a_element_desc,
                        B=b_element_desc,
                        C=c_element_desc,
                        a_elem_op=element_op,
                        b_elem_op=element_op,
                        epilogue_functor=c_element_op,
                        gemm_specialization=gemm_spec,
                        tile_desc=tile_desc,
                        a_block_transfer=block_desc,
                        b_block_transfer=block_desc,
                        c_block_transfer=c_block_desc,
                        ds_dtype=ds_dtype,
                        e_dtype=e_dtype,
                        loop_scheduler=loop_scheduler,
                        pipeline=pipeline
                    )
                    manifest.append(new_operation)
                    operations.append(new_operation)
    return operations


def CreateGemmRRRPermOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.GemmPermute
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    ds_dtype = [library.DataType.f16]
    e_dtype = library.DataType.f16
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 2, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 2, 32, 32, 2, 1),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 2, 32, 32, 1, 2),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
    ]

    b_block_descriptions = [
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([8, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([16, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
    ]
    a_block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        a_block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            a_block_transfer = [4, 64, 1]
            # TODO:figure out the last dimension
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 1)
        if t.block_size == 128:
            a_block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 1)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 1)
        if t.block_size == 64:
            a_block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 1)

        assert (
            a_block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        a_block_descriptions.append(
            gemm.BlockTransferDesc(a_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]

    loop_schedulers = ["ck::LoopScheduler::Default", "ck::LoopScheduler::Interwave"]
    pipelines = ["ck::PipelineVersion::v1", "ck::PipelineVersion::v2"]

    operations = []
    for loop_scheduler in loop_schedulers:
        for pipeline in pipelines:
            if pipeline == "ck::PipelineVersion::v2" and loop_scheduler == "ck::LoopScheduler::Interwave":
                continue
            for gemm_spec in gemm_specialization:
                for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
                    tile_descriptions,
                    a_block_descriptions,
                    b_block_descriptions,
                    c_block_descriptions,
                ):
                    new_operation = gemm.GemmOperation(
                        operation_kind=operation_kind,
                        extra_kind=c_element_op,
                        xdl_op_type=gemm.XdlOpType.DeviceGemmBiasCPermute_Xdl,
                        A=a_element_desc,
                        B=b_element_desc,
                        C=c_element_desc,
                        a_elem_op=element_op,
                        b_elem_op=element_op,
                        epilogue_functor=c_element_op,
                        gemm_specialization=gemm_spec,
                        tile_desc=tile_desc,
                        a_block_transfer=a_block_desc,
                        b_block_transfer=b_block_desc,
                        c_block_transfer=c_block_desc,
                        ds_dtype=ds_dtype,
                        e_dtype=e_dtype,
                        loop_scheduler=loop_scheduler,
                        pipeline=pipeline
                    )
                    manifest.append(new_operation)
                    operations.append(new_operation)
    return operations


def CreateGemmRCRm2n3PermOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.GemmPermuteM2N3
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    ds_dtype = [library.DataType.f16]
    e_dtype = library.DataType.f16
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        # gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
            # TODO:figure out the last dimension
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8)

        assert (
            block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]

    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, block_desc, c_block_desc in zip(
            tile_descriptions, block_descriptions, c_block_descriptions
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=c_element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedContractionMultipleD_Xdl_CShuffle,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=c_element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=block_desc,
                b_block_transfer=block_desc,
                c_block_transfer=c_block_desc,
                ds_dtype=ds_dtype,
                e_dtype=e_dtype,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateGemmRCRm3n2PermOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.GemmPermuteM3N2
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    ds_dtype = [library.DataType.f16]
    e_dtype = library.DataType.f16
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        # gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
            # TODO:figure out the last dimension
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 1)
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 1)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 1)
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 1)

        assert (
            block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]

    
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, block_desc, c_block_desc in zip(
            tile_descriptions, block_descriptions, c_block_descriptions
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=c_element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedContractionMultipleD_Xdl_CShuffle,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=c_element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=block_desc,
                b_block_transfer=block_desc,
                c_block_transfer=c_block_desc,
                ds_dtype=ds_dtype,
                e_dtype=e_dtype,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmRCRPermOperator(manifest):
    operation_kind = library.GemmKind.BatchGemmPermute
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8)

        assert (
            block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)

    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, block_desc, c_block_desc in zip(
            tile_descriptions, block_descriptions, c_block_descriptions
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmCPermuteXdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=block_desc,
                b_block_transfer=block_desc,
                c_block_transfer=c_block_desc,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmSoftmaxBmmOperator(
    manifest,
    operation_kind=library.GemmKind.BatchGemmSoftmaxGemm,
    xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle,
):
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    tile_descriptions = [
        gemm.AttnTileDesc(256, 256, 128, 32, 64, 32, 8, 8, 2, 32, 32, 2, 4, 2),
        gemm.AttnTileDesc(256, 256, 128, 32, 128, 32, 8, 8, 2, 32, 32, 2, 4, 4),
        gemm.AttnTileDesc(256, 128, 256, 32, 64, 32, 8, 8, 2, 32, 32, 1, 8, 2),
        gemm.AttnTileDesc(256, 128, 256, 32, 128, 32, 8, 8, 2, 32, 32, 1, 8, 4),
        gemm.AttnTileDesc(256, 128, 128, 64, 64, 32, 8, 8, 2, 32, 32, 1, 4, 2),
        gemm.AttnTileDesc(256, 128, 128, 32, 64, 32, 8, 8, 2, 32, 32, 1, 4, 2),
        gemm.AttnTileDesc(256, 128, 128, 64, 128, 32, 8, 8, 2, 32, 32, 1, 4, 4),
        gemm.AttnTileDesc(256, 128, 128, 32, 128, 32, 8, 8, 2, 32, 32, 1, 4, 4),
        gemm.AttnTileDesc(256, 64, 256, 32, 128, 32, 8, 8, 2, 16, 16, 1, 16, 8),
        gemm.AttnTileDesc(256, 64, 256, 32, 64, 32, 8, 8, 2, 16, 16, 1, 16, 4),
        gemm.AttnTileDesc(256, 64, 256, 64, 128, 32, 8, 8, 2, 16, 16, 1, 16, 8),
        gemm.AttnTileDesc(256, 64, 256, 64, 64, 32, 8, 8, 2, 16, 16, 1, 16, 4),
        gemm.AttnTileDesc(256, 128, 128, 64, 128, 32, 8, 8, 2, 32, 32, 1, 4, 4),
        gemm.AttnTileDesc(256, 128, 64, 32, 128, 32, 8, 8, 2, 32, 32, 1, 2, 4),
    ]

    block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 0),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 0),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 0),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
    ]
    c_block_descriptions, b1_block_descriptions = [], []
    for i in range(len(tile_descriptions)):
        if i in [0, 2, 4, 5, 9, 11]:
            block_transfer = [16, 16, 1]
        else:
            block_transfer = [8, 32, 1]
        b1_block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [0, 2, 1], [0, 2, 1], 1, 4, 2, 0)
        )

        if i in [8, 10]:
            c_block_transfer = gemm.CBlockTransferDesc(1, 8, [1, 16, 1, 16], 8)
        else:
            c_shuffle = 4 if i in [9, 11] else 2
            c_block_transfer = gemm.CBlockTransferDesc(1, c_shuffle, [1, 32, 1, 8], 8)

        c_block_descriptions.append(c_block_transfer)

    gemm_specialization = []
    for i in range(len(tile_descriptions)):
        if i < 12:
            gemm_specialization.append(gemm.GemmSpecialization.GemmDefault)
        else:
            gemm_specialization.append(gemm.GemmSpecialization.MNOPadding)

    operations = []
    for tile_desc, block_desc, b1_block_desc, c_block_desc, gemm_spec in zip(
        tile_descriptions,
        block_descriptions,
        b1_block_descriptions,
        c_block_descriptions,
        gemm_specialization,
    ):
        new_operation = gemm.GemmOperation(
            operation_kind=operation_kind,
            extra_kind=element_op,
            xdl_op_type=xdl_op_type,
            A=a_element_desc,
            B=b_element_desc,
            C=c_element_desc,
            a_elem_op=element_op,
            b_elem_op=element_op,
            epilogue_functor=element_op,
            gemm_specialization=gemm_spec,
            tile_desc=tile_desc,
            a_block_transfer=block_desc,
            b_block_transfer=block_desc,
            b1_block_transfer=b1_block_desc,
            c_block_transfer=c_block_desc,
        )
        manifest.append(new_operation)
        operations.append(new_operation)
    return operations


def CreateBmmSoftmaxBmmPermOperator(
    manifest,
    operation_kind=library.GemmKind.BatchGemmSoftmaxGemmPermute,
    xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle,
    causal_mask=None,
):
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    tile_descriptions = [
        gemm.AttnTileDesc(256, 256, 128, 32, 64, 32, 8, 8, 2, 32, 32, 2, 4, 2),
        gemm.AttnTileDesc(256, 256, 128, 32, 128, 32, 8, 8, 2, 32, 32, 2, 4, 4),
        gemm.AttnTileDesc(256, 128, 256, 32, 64, 32, 8, 8, 2, 32, 32, 1, 8, 2),
        gemm.AttnTileDesc(256, 128, 256, 32, 128, 32, 8, 8, 2, 32, 32, 1, 8, 4),
        gemm.AttnTileDesc(256, 128, 128, 64, 64, 32, 8, 8, 2, 32, 32, 1, 4, 2),
        gemm.AttnTileDesc(256, 128, 128, 32, 64, 32, 8, 8, 2, 32, 32, 1, 4, 2),
        gemm.AttnTileDesc(256, 128, 128, 64, 128, 32, 8, 8, 2, 32, 32, 1, 4, 4),
        gemm.AttnTileDesc(256, 128, 128, 32, 128, 32, 8, 8, 2, 32, 32, 1, 4, 4),
        gemm.AttnTileDesc(256, 64, 256, 32, 128, 32, 8, 8, 2, 16, 16, 1, 16, 8),
        gemm.AttnTileDesc(256, 64, 256, 32, 64, 32, 8, 8, 2, 16, 16, 1, 16, 4),
        gemm.AttnTileDesc(256, 64, 256, 64, 128, 32, 8, 8, 2, 16, 16, 1, 16, 8),
        gemm.AttnTileDesc(256, 64, 256, 64, 64, 32, 8, 8, 2, 16, 16, 1, 16, 4),
        gemm.AttnTileDesc(256, 128, 128, 64, 128, 32, 8, 8, 2, 32, 32, 1, 4, 4),
        gemm.AttnTileDesc(256, 128, 64, 32, 128, 32, 8, 8, 2, 32, 32, 1, 2, 4),
        # for MNKOPadding
        gemm.AttnTileDesc(256, 128, 128, 64, 128, 32, 8, 8, 2, 32, 32, 1, 4, 4),
        gemm.AttnTileDesc(256, 128, 64, 32, 128, 32, 8, 8, 2, 32, 32, 1, 2, 4),
    ]

    block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 0),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 0),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 0),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
        # for MNKOPadding
        gemm.BlockTransferDesc([8, 32, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 0),
        gemm.BlockTransferDesc([4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1),
    ]
    causal_mask_flag = 0
    if causal_mask is not None:
        causal_mask_flag = 1 if library.TensorOperationTag[causal_mask] == "True" else 0

    c_block_descriptions, b1_block_descriptions = [], []
    for i in range(len(tile_descriptions)):
        if i in [0, 2, 4, 5, 9, 11]:
            block_transfer = [16, 16, 1]
        else:
            block_transfer = [8, 32, 1]
        b1_block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [0, 2, 1], [0, 2, 1], 1, 4, 2, 0)
        )

        if i in [8, 10]:
            c_block_transfer = gemm.MaskedCBlockTransferDesc(
                1, 8, [1, 16, 1, 16], 8, causal_mask_flag
            )
        else:
            c_shuffle = 4 if i in [9, 11] else 2
            c_block_transfer = gemm.MaskedCBlockTransferDesc(
                1, c_shuffle, [1, 32, 1, 8], 8, causal_mask_flag
            )

        c_block_descriptions.append(c_block_transfer)

    gemm_specialization = []
    for i in range(len(tile_descriptions)):
        if i < 12:
            gemm_specialization.append(gemm.GemmSpecialization.GemmDefault)
        elif i in [12, 13]:
            gemm_specialization.append(gemm.GemmSpecialization.MNOPadding)
        else:
            gemm_specialization.append(gemm.GemmSpecialization.MNKOPadding)

    operations = []
    extra_op = element_op if causal_mask_flag == 0 else causal_mask
    for tile_desc, block_desc, b1_block_desc, c_block_desc, gemm_spec in zip(
        tile_descriptions,
        block_descriptions,
        b1_block_descriptions,
        c_block_descriptions,
        gemm_specialization,
    ):
        new_operation = gemm.GemmOperation(
            operation_kind=operation_kind,
            extra_kind=extra_op,
            xdl_op_type=xdl_op_type,
            A=a_element_desc,
            B=b_element_desc,
            C=c_element_desc,
            a_elem_op=element_op,
            b_elem_op=element_op,
            epilogue_functor=element_op,
            gemm_specialization=gemm_spec,
            tile_desc=tile_desc,
            a_block_transfer=block_desc,
            b_block_transfer=block_desc,
            b1_block_transfer=b1_block_desc,
            c_block_transfer=c_block_desc,
        )
        manifest.append(new_operation)
        operations.append(new_operation)
    return operations


def CreateBmmRRROperator(manifest):
    operation_kind = library.GemmKind.BatchGemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 4, 8, 0, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 4, 8, 0, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 4, 8, 0, 32, 32, 1, 2),
        gemm.TileDesc(128, 32, 256, 4, 8, 0, 32, 32, 1, 4),
        gemm.TileDesc(128, 32, 128, 4, 8, 0, 32, 32, 1, 2),
        gemm.TileDesc(128, 32, 64, 4, 8, 0, 32, 32, 1, 1),
        gemm.TileDesc(64, 32, 32, 4, 8, 0, 32, 32, 1, 1),
        gemm.TileDesc(128, 16, 256, 4, 8, 0, 16, 16, 1, 8),
        gemm.TileDesc(128, 16, 128, 4, 8, 0, 16, 16, 1, 4),
        gemm.TileDesc(128, 16, 64, 4, 8, 0, 16, 16, 1, 2),
        gemm.TileDesc(128, 16, 32, 4, 8, 0, 16, 16, 1, 1),
        gemm.TileDesc(64, 16, 16, 4, 8, 0, 16, 16, 1, 1),
    ]

    a_block_descriptions = []
    for t in tile_descriptions:
        a_block_transfer = -1
        if t.block_size == 256:
            a_block_transfer = [4, 64, 1]
        if t.block_size == 128 and t.m_per_block != 16:
            a_block_transfer = [4, 32, 1]
        if t.block_size == 64 or t.m_per_block == 16:
            a_block_transfer = [4, 16, 1]

        assert (
            a_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        a_block_descriptions.append(
            gemm.BlockTransferDesc(
                a_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
            )
        )
    b_block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 8, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 16, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 8, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
        gemm.BlockTransferDesc([4, 16, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
    ]
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc in zip(
            tile_descriptions, a_block_descriptions, b_block_descriptions
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmXdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmRRRBillinearOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.BatchGemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 2, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 2, 32, 32, 2, 1),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 2, 32, 32, 1, 2),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
    ]

    a_block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        a_block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            a_block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128 and t.n_per_block != 64:
            a_block_transfer = [4, 32, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
        if t.block_size == 128 and t.n_per_block == 64:
            a_block_transfer = [4, 32, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)

        assert (
            a_block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        a_block_descriptions.append(
            gemm.BlockTransferDesc(
                a_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
            )
        )
        c_block_descriptions.append(c_block_transfer)
    b_block_descriptions = [
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([8, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([16, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
    ]
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    ds_dtype = [library.DataType.f16]
    ds_layout = [library.LayoutType.RowMajor]
    e_dtype = library.DataType.f16
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=c_element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmMultiD_Xdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=c_element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                c_block_transfer=c_block_desc,
                ds_dtype=ds_dtype,
                ds_layout=ds_layout,
                e_dtype=e_dtype,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmCCRBillinearOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.BatchGemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 2, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 2, 8, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 2, 8, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 2, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 2, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 2, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 2, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 2, 8, 32, 32, 1, 2),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
    ]

    b_block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        b_block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            b_block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128 and t.n_per_block != 64:
            b_block_transfer = [4, 32, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
        if t.block_size == 128 and t.n_per_block == 64:
            b_block_transfer = [4, 32, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)

        assert (
            b_block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        b_block_descriptions.append(
            gemm.BlockTransferDesc(
                b_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
            )
        )
        c_block_descriptions.append(c_block_transfer)
    a_block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([8, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([16, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
    ]
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    ds_dtype = [library.DataType.f16]
    ds_layout = [library.LayoutType.RowMajor]
    e_dtype = library.DataType.f16
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=c_element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmMultiD_Xdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=c_element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                c_block_transfer=c_block_desc,
                ds_dtype=ds_dtype,
                ds_layout=ds_layout,
                e_dtype=e_dtype,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmCRRBillinearOperator(manifest, c_element_op):
    operation_kind = library.GemmKind.BatchGemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 2, 2, 32, 32, 4, 2),
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 2, 2, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 2, 2, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 2, 2, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 2, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 2, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 2, 2, 32, 32, 2, 1),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 2, 2, 32, 32, 1, 2),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
    ]

    b_block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        b_block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            b_block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128 and t.n_per_block != 64:
            b_block_transfer = [4, 32, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
        if t.block_size == 128 and t.n_per_block == 64:
            b_block_transfer = [4, 32, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)

        assert (
            b_block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        b_block_descriptions.append(
            gemm.BlockTransferDesc(
                b_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
            )
        )
        c_block_descriptions.append(c_block_transfer)
    a_block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([8, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([16, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
    ]
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    ds_dtype = [library.DataType.f16]
    ds_layout = [library.LayoutType.RowMajor]
    e_dtype = library.DataType.f16
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=c_element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmMultiD_Xdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=c_element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                c_block_transfer=c_block_desc,
                ds_dtype=ds_dtype,
                ds_layout=ds_layout,
                e_dtype=e_dtype,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmRRRPermOperator(manifest):
    operation_kind = library.GemmKind.BatchGemmPermute
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 2, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 2, 32, 32, 2, 1),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 2, 32, 32, 1, 2),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
    ]

    b_block_descriptions = [
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([8, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([16, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
    ]
    a_block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        a_block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            a_block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            a_block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            if t.n_per_block == 64:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)

        assert (
            a_block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        a_block_descriptions.append(
            gemm.BlockTransferDesc(a_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)

    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmCPermuteXdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                c_block_transfer=c_block_desc,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmCCROperator(manifest):
    operation_kind = library.GemmKind.BatchGemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 4, 8, 0, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 4, 8, 0, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 4, 8, 0, 32, 32, 1, 2),
    ]

    b_block_descriptions = []
    for t in tile_descriptions:
        b_block_transfer = -1
        if t.block_size == 256:
            b_block_transfer = [4, 64, 1]
        if t.block_size == 128 and t.m_per_block != 16:
            b_block_transfer = [4, 32, 1]

        assert (
            b_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size "
            + str(t.block_size)
        )
        b_block_descriptions.append(
            gemm.BlockTransferDesc(
                b_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
            )
        )
    a_block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
    ]
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc in zip(
            tile_descriptions, a_block_descriptions, b_block_descriptions
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmXdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateBmmCRROperator(manifest):
    operation_kind = library.GemmKind.BatchGemm
    a_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.ColumnMajor
    )
    b_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    c_element_desc = library.TensorDesc(
        library.DataType.f16, library.LayoutType.RowMajor
    )
    element_op = library.TensorOperation.PassThrough
    # 0 indicates not print
    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 4, 8, 0, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 4, 8, 0, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 4, 8, 0, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 4, 8, 0, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 4, 8, 0, 32, 32, 1, 2),
    ]

    a_block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
    ]
    b_block_descriptions = [
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1, True),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1, True),
    ]
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc in zip(
            tile_descriptions, a_block_descriptions, b_block_descriptions
        ):
            new_operation = gemm.GemmOperation(
                operation_kind=operation_kind,
                extra_kind=element_op,
                xdl_op_type=gemm.XdlOpType.DeviceBatchedGemmXdl,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
            )
            manifest.append(new_operation)
            operations.append(new_operation)
    return operations


def CreateSoftmaxOperator(manifest, rank=3):
    operation_kind = library.OperationKind.Softmax
    in_dtype = library.DataType.f16
    out_dtype = library.DataType.f16
    # 0 indicates not print
    tile_descriptions = [
        softmax.TileDesc(256, 8, 32, 1, 8, 1, 1, 1),
        softmax.TileDesc(256, 8, 32, 1, 8, 1, 8, 8),
        softmax.TileDesc(256, 4, 64, 1, 8, 1, 8, 8),
        softmax.TileDesc(256, 2, 128, 1, 8, 1, 8, 8),
        softmax.TileDesc(256, 2, 128, 1, 16, 1, 8, 8),
        softmax.TileDesc(256, 2, 128, 1, 32, 1, 8, 8),
        softmax.TileDesc(256, 1, 256, 1, 8, 1, 8, 8),
        softmax.TileDesc(256, 1, 256, 1, 16, 1, 8, 8),
        softmax.TileDesc(256, 1, 256, 1, 32, 1, 8, 8),
    ]

    operations = []
    for tile_desc in tile_descriptions:
        new_operation = softmax.SoftmaxOperation(
            operation_kind=operation_kind,
            extra_kind=rank,
            In=in_dtype,
            Out=out_dtype,
            Rank=rank,
            NumReduceDim=1,
            tile_desc=tile_desc,
        )
        manifest.append(new_operation)
        operations.append(new_operation)
    return operations


def CreateLayerNormOperator(manifest, rank=2):
    operation_kind = library.OperationKind.LayerNorm
    in_dtype = library.DataType.f16
    out_dtype = library.DataType.f16
    # 0 indicates not print
    tile_descriptions = [
        layernorm.TileDesc(128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        layernorm.TileDesc(256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        layernorm.TileDesc(512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        layernorm.TileDesc(1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        layernorm.TileDesc(256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1),
        layernorm.TileDesc(256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1),
        layernorm.TileDesc(64, 1, 64, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(128, 1, 128, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(128, 1, 128, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(128, 1, 128, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(256, 1, 256, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(256, 1, 256, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(256, 1, 256, 2, 16, 1, 8, 1, 8, 1, 8, 8, 2),
        layernorm.TileDesc(256, 1, 256, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(512, 1, 512, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(512, 1, 512, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(1024, 1, 1024, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        layernorm.TileDesc(1024, 1, 1024, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
    ]

    operations = []
    for tile_desc in tile_descriptions:
        new_operation = layernorm.LayerNormOperation(
            operation_kind=operation_kind,
            extra_kind=rank,
            In=in_dtype,
            Out=out_dtype,
            Rank=rank,
            NumReduceDim=1,
            tile_desc=tile_desc,
        )
        manifest.append(new_operation)
        operations.append(new_operation)
    return operations


def CreateGroupNormOperator(manifest, rank=5):
    operation_kind = library.OperationKind.GroupNorm
    in_dtype = library.DataType.f16
    out_dtype = library.DataType.f16
    # 0 indicates not print
    tile_descriptions = [
        groupnorm.TileDesc(128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        groupnorm.TileDesc(256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        groupnorm.TileDesc(512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        groupnorm.TileDesc(1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        groupnorm.TileDesc(256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1),
        groupnorm.TileDesc(256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1),
        groupnorm.TileDesc(64, 1, 64, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(128, 1, 128, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(128, 1, 128, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(128, 1, 128, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(256, 1, 256, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(256, 1, 256, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(256, 1, 256, 2, 16, 1, 8, 1, 8, 1, 8, 8, 2),
        groupnorm.TileDesc(256, 1, 256, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(512, 1, 512, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(512, 1, 512, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(1024, 1, 1024, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1),
        groupnorm.TileDesc(1024, 1, 1024, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1),
    ]

    operations = []
    for tile_desc in tile_descriptions:
        new_operation = groupnorm.GroupNormOperation(
            operation_kind=operation_kind,
            extra_kind=rank,
            In=in_dtype,
            Out=out_dtype,
            Rank=rank,
            NumReduceDim=3,
            tile_desc=tile_desc,
        )
        manifest.append(new_operation)
        operations.append(new_operation)
    return operations


def GenerateTensorOp(manifest):
    # Conv2d
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.PassThrough,
    )
    # Conv2dBias
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.Add,
        library.MemoryDataOperation.MemorySet,
    )
    # Conv2dBiasRelu
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.AddRelu,
        library.MemoryDataOperation.MemorySet,
    )
    # Conv2dBiasAdd
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.AddAdd,
    )
    # Conv2dBiasReluAdd
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.AddReluAdd,
    )
    # Conv2dBiasAddRelu
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.AddAddRelu,
    )
    # Conv2dBiasSigmoid
    CreateConv2dFwdOperator(
        manifest,
        library.Conv2dKind.GroupConv2dBiasRelu,
        library.TensorOperation.AddSigmoid,
        library.MemoryDataOperation.MemorySet,
    )
    # GemmRRR
    CreateGemmRRROperator(manifest)
    # GemmRCR
    CreateGemmRCROperator(manifest)
    # GemmRCRBias
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.Add)
    # GemmRCRBiasRelu
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddRelu)
    # GemmRCRBiasTanh
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddTanh)
    # GemmRCRBiasTanh
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddFastGelu)
    # GemmRCRBiasHardswish
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddHardswish)
    # GemmRCRBiasSwish
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddSwish)
    # GemmRCRBiasSigmoid
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddSigmoid)
    # GemmRCRBiasAdd
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddAdd)
    # GemmRCRBiasMul
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddMul)
    # GemmRCRBiasMul
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddMulTanh)
    # GemmRCRBiasAddRelu
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddAddRelu)
    # GemmRCRBiasAddAddRelu
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddAddAdd)
    # GemmRCRBiasAddAddRelu
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddAddAddRelu)
    # GemmRCRBiasSigmoidMul
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddSigmoidMul)
    # GemmRCRBiasSigmoidMulTanh
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddSigmoidMulTanh)
    # GemmRCRBiasMulAdd
    CreateGemmRCRBilinearOperator(manifest, library.TensorOperation.AddMulAdd)
    # BmmRCR
    CreateBmmRCROperator(manifest)
    # BmmRRR
    CreateBmmRRROperator(manifest)
    # BmmRRRAdd
    CreateBmmRRRBillinearOperator(manifest, library.TensorOperation.Add)
    # BmmCRRAdd
    CreateBmmCRRBillinearOperator(manifest, library.TensorOperation.Add)
    # BmmCRRAdd
    CreateBmmCCRBillinearOperator(manifest, library.TensorOperation.Add)
    # BmmCCR
    CreateBmmCCROperator(manifest)
    # BmmCRR
    CreateBmmCRROperator(manifest)
    # BmmRCR-Permute
    CreateBmmRCRPermOperator(manifest)
    # BmmRRR-Permute
    CreateBmmRRRPermOperator(manifest)
    # GemmBiasRCR-Permute
    CreateGemmRCRPermOperator(manifest, library.TensorOperation.Add)
    CreateGemmRCRm2n3PermOperator(manifest, library.TensorOperation.Add)
    CreateGemmRCRm2n3PermOperator(manifest, library.TensorOperation.PassThrough)
    CreateGemmRCRm3n2PermOperator(manifest, library.TensorOperation.Add)
    # GemmBiasRRR-Permute
    CreateGemmRRRPermOperator(manifest, library.TensorOperation.Add)
    # Bmm-Softmax-Bmm
    CreateBmmSoftmaxBmmOperator(manifest)
    # Attention (Bmm-Softmax-Bmm-Permute)
    CreateBmmSoftmaxBmmPermOperator(manifest)
    # Attention with Causal Mask
    CreateBmmSoftmaxBmmPermOperator(
        manifest, causal_mask=library.TensorOperation.CausalMask
    )
    # Softmax
    for rank in range(2, 5):
        CreateSoftmaxOperator(manifest, rank=rank)

    CreateLayerNormOperator(manifest, rank=2)
    CreateGroupNormOperator(manifest, rank=5)


def GenerateGFX908(manifest, rocm_version):
    GenerateTensorOp(manifest)

def GenerateGFX90A(manifest, rocm_version):
    GenerateTensorOp(manifest)

def GenerateGFX940(manifest, rocm_version):
    GenerateTensorOp(manifest)

def GenerateGFX941(manifest, rocm_version):
    GenerateTensorOp(manifest)

def GenerateGFX942(manifest, rocm_version):
    GenerateTensorOp(manifest)
