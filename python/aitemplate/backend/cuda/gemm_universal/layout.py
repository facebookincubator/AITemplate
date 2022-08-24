# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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
