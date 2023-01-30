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
GeMM layout classes.
"""

from dataclasses import dataclass

# pylint: disable=C0415


@dataclass
class Layout:
    m = "M"
    n = "N"
    k = "K"


@dataclass
class RCR(Layout):
    """
    Layout: A[RowMajor], B[ColumnMajor], C[RowMajor]
    """

    cutlass_layout_a = "cutlass::layout::RowMajor"
    cutlass_layout_b = "cutlass::layout::ColumnMajor"
    cutlass_layout_c = "cutlass::layout::RowMajor"
    stride_a = "K"
    stride_b = "K"
    stride_c = "N"

    args_parser = """
  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = N;
  int64_t b_dim1 = K;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""

    @staticmethod
    def fproc_op(op):
        import cutlass_lib

        row_major = cutlass_lib.library.LayoutType.RowMajor
        op.C.layout = row_major

    @staticmethod
    def fcond_op(op):
        import cutlass_lib

        row_major = cutlass_lib.library.LayoutType.RowMajor
        col_major = cutlass_lib.library.LayoutType.ColumnMajor
        return op.A.layout == row_major and op.B.layout == col_major

    @staticmethod
    def cutlass_lib_layouts():
        """
        return [layout_a, layout_b, layout_c] in the form of cutlass_lib definitions
        """
        import cutlass_lib

        return [
            cutlass_lib.library.LayoutType.RowMajor,
            cutlass_lib.library.LayoutType.ColumnMajor,
            cutlass_lib.library.LayoutType.RowMajor,
        ]


@dataclass
class RRR(Layout):
    """
    Layout: A[RowMajor], B[RowMajor], C[RowMajor]
    """

    cutlass_layout_a = "cutlass::layout::RowMajor"
    cutlass_layout_b = "cutlass::layout::RowMajor"
    cutlass_layout_c = "cutlass::layout::RowMajor"
    stride_a = "K"
    stride_b = "N"
    stride_c = "N"

    args_parser = """
  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = K;
  int64_t b_dim1 = N;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""

    @staticmethod
    def fproc_op(op):
        import cutlass_lib

        row_major = cutlass_lib.library.LayoutType.RowMajor
        op.C.layout = row_major

    @staticmethod
    def fcond_op(op):
        import cutlass_lib

        row_major = cutlass_lib.library.LayoutType.RowMajor
        return op.A.layout == row_major and op.B.layout == row_major

    @staticmethod
    def cutlass_lib_layouts():
        """
        return [layout_a, layout_b, layout_c] in the form of cutlass_lib definitions
        """
        import cutlass_lib

        return [
            cutlass_lib.library.LayoutType.RowMajor,
            cutlass_lib.library.LayoutType.RowMajor,
            cutlass_lib.library.LayoutType.RowMajor,
        ]
