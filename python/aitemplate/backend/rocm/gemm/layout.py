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

    ck_layout_a = "ck::tensor_layout::gemm::RowMajor"
    ck_layout_b = "ck::tensor_layout::gemm::ColumnMajor"
    ck_layout_c = "ck::tensor_layout::gemm::RowMajor"
    stride_a = "K"
    stride_b = "K"
    stride_c = "N"

    args_parse = """
  int64_t M = std::stoi(argv[1]);
  int64_t N = std::stoi(argv[2]);
  int64_t K = std::stoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);
  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = N;
  int64_t b_dim1 = K;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""

    @staticmethod
    def fproc_op(op):
        import ck_lib

        row_major = ck_lib.library.LayoutType.RowMajor
        op.C.layout = row_major

    @staticmethod
    def fcond_op(op):
        import ck_lib

        row_major = ck_lib.library.LayoutType.RowMajor
        col_major = ck_lib.library.LayoutType.ColumnMajor
        return op.A.layout == row_major and op.B.layout == col_major

    @staticmethod
    def ck_lib_layouts():
        """
        return [layout_a, layout_b, layout_c] in the form of ck_lib definitions
        """
        import ck_lib

        return [
            ck_lib.library.LayoutType.RowMajor,
            ck_lib.library.LayoutType.ColumnMajor,
            ck_lib.library.LayoutType.RowMajor,
        ]


@dataclass
class RRR(Layout):
    """
    Layout: A[RowMajor], B[RowMajor], C[RowMajor]
    """

    ck_layout_a = "ck::tensor_layout::gemm::RowMajor"
    ck_layout_b = "ck::tensor_layout::gemm::RowMajor"
    ck_layout_c = "ck::tensor_layout::gemm::RowMajor"
    stride_a = "K"
    stride_b = "N"
    stride_c = "N"

    args_parse = """
  int64_t M = std::stoi(argv[1]);
  int64_t N = std::stoi(argv[2]);
  int64_t K = std::stoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);
  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = K;
  int64_t b_dim1 = N;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""

    @staticmethod
    def fproc_op(op):
        import ck_lib

        row_major = ck_lib.library.LayoutType.RowMajor
        op.C.layout = row_major

    @staticmethod
    def fcond_op(op):
        import ck_lib

        row_major = ck_lib.library.LayoutType.RowMajor
        return op.A.layout == row_major and op.B.layout == row_major

    @staticmethod
    def ck_lib_layouts():
        """
        return [layout_a, layout_b, layout_c] in the form of ck_lib definitions
        """
        import ck_lib

        return [
            ck_lib.library.LayoutType.RowMajor,
            ck_lib.library.LayoutType.RowMajor,
            ck_lib.library.LayoutType.RowMajor,
        ]


@dataclass
class CCR(Layout):
    """
    Layout: A[ColumnMajor], B[ColumnMajor], C[RowMajor]
    """

    ck_layout_a = "ck::tensor_layout::gemm::ColumnMajor"
    ck_layout_b = "ck::tensor_layout::gemm::ColumnMajor"
    ck_layout_c = "ck::tensor_layout::gemm::RowMajor"
    stride_a = "M"
    stride_b = "K"
    stride_c = "N"

    args_parse = """
  int64_t M = std::stoi(argv[1]);
  int64_t N = std::stoi(argv[2]);
  int64_t K = std::stoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);
  int64_t a_dim0 = K;
  int64_t a_dim1 = M;
  int64_t b_dim0 = N;
  int64_t b_dim1 = K;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""

    @staticmethod
    def fproc_op(op):
        import ck_lib

        row_major = ck_lib.library.LayoutType.RowMajor
        op.C.layout = row_major

    @staticmethod
    def fcond_op(op):
        import ck_lib

        col_major = ck_lib.library.LayoutType.ColumnMajor
        return op.A.layout == col_major and op.B.layout == col_major

    @staticmethod
    def ck_lib_layouts():
        """
        return [layout_a, layout_b, layout_c] in the form of ck_lib definitions
        """
        import ck_lib

        return [
            ck_lib.library.LayoutType.ColumnMajor,
            ck_lib.library.LayoutType.ColumnMajor,
            ck_lib.library.LayoutType.RowMajor,
        ]


@dataclass
class CRR(Layout):
    """
    Layout: A[ColumnMajor], B[RowMajor], C[RowMajor]
    """

    ck_layout_a = "ck::tensor_layout::gemm::ColumnMajor"
    ck_layout_b = "ck::tensor_layout::gemm::RowMajor"
    ck_layout_c = "ck::tensor_layout::gemm::RowMajor"
    stride_a = "M"
    stride_b = "N"
    stride_c = "N"

    args_parse = """
  int64_t M = std::stoi(argv[1]);
  int64_t N = std::stoi(argv[2]);
  int64_t K = std::stoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);
  int64_t a_dim0 = K;
  int64_t a_dim1 = M;
  int64_t b_dim0 = K;
  int64_t b_dim1 = N;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;
"""

    @staticmethod
    def fproc_op(op):
        import ck_lib

        row_major = ck_lib.library.LayoutType.RowMajor
        op.C.layout = row_major

    @staticmethod
    def fcond_op(op):
        import ck_lib

        row_major = ck_lib.library.LayoutType.RowMajor
        col_major = ck_lib.library.LayoutType.ColumnMajor
        return op.A.layout == col_major and op.B.layout == row_major

    @staticmethod
    def ck_lib_layouts():
        """
        return [layout_a, layout_b, layout_c] in the form of ck_lib definitions
        """
        import ck_lib

        return [
            ck_lib.library.LayoutType.ColumnMajor,
            ck_lib.library.LayoutType.RowMajor,
            ck_lib.library.LayoutType.RowMajor,
        ]
