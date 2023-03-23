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


from aitemplate.compiler.base import Tensor
from aitemplate.compiler.ops.gemm_universal.bmm import (
    is_valid_inputs as bmm_is_valid_inputs,
)
from aitemplate.compiler.ops.gemm_universal.bmm_xxx import (
    bmm_ccc,
    bmm_ccr,
    bmm_crc,
    bmm_crr,
    bmm_rcc,
    bmm_rcr,
    bmm_rrc,
    bmm_rrr,
    bmm_xxx,
)
from aitemplate.compiler.tensor_accessor import TensorAccessor


class bmm_xxx_add(bmm_xxx):
    """Batch GEMM specialization with Add.
    C can be the same size as the output or be broadcast as bias.
    """

    def __init__(self, a_layout, b_layout, c_layout):
        super().__init__(a_layout, b_layout, c_layout)
        self._attrs["op"] = f"bmm_{a_layout}{b_layout}{c_layout}_add"
        self._attrs["has_d"] = True

    def __call__(self, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """Call bmm_rrr_add with tensors a, b, c"""
        output = super().__call__(a, b)
        self._attrs["inputs"].append(c)
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        return output

    def is_valid_inputs_unspecialized(self, A: Tensor, B: Tensor, C: Tensor):
        # For base bmm_xxx_add class this method can't be static,
        # since the class doesn't know about the layout (the object does).
        output_shapes = bmm_xxx(
            self.a_layout, self.b_layout, self.c_layout
        )._infer_shapes(A, B)
        c_shapes = C.shape()
        return bmm_is_valid_inputs(output_shapes, c_shapes)

    @classmethod
    def is_valid_inputs(cls, A: Tensor, B: Tensor, C: Tensor):
        """
        This method should only be called from subclasses of bmm_xxx_add, since
        _SpecializedBase is defined there. For the parent class bmm_xxx_add itself
        call is_valid_inputs_unspecialized instead.
        """
        if not hasattr(cls, "_SpecializedBase"):
            raise NotImplementedError(
                "Call bmm_xxx_add.is_valid_inputs_unspecialized instead of bmm_xxx_add.is_valid_inputs. The latter is only defined for child classes of bmm_xxx_add."
            )
        output_shapes = cls._SpecializedBase()._infer_shapes(A, B)
        c_shapes = C.shape()
        return bmm_is_valid_inputs(output_shapes, c_shapes)


class bmm_crr_add(bmm_xxx_add):
    """Batch GEMM specialization for A[ColMajor], B[RowMajor], C[RowMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()
        D_pt = torch.randn(B, M, N).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        Y_pt = torch.bmm(XT, W_pt)
        Y_pt = Y_pt + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor in shape (B, K, M)
        b : Tensor
            Tensor in shape (B, K, N)
        c : Tensor
            Tensor in shape (B, M, N)

        Returns
        -------
        Tensor
            Tensor in shape (B, M, N)

    """

    _SpecializedBase = bmm_crr

    def __init__(self):
        """Constructor for bmm_crr_add"""
        super().__init__("c", "r", "r")


class bmm_rcr_add(bmm_xxx_add):
    """Batch GEMM specialization for A[RowMajor], B[ColMajor], C[RowMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()
        D_pt = torch.randn(B, M, N).cuda().half()

        WT = torch.transpose(W_pt, 2, 1)
        Y_pt = torch.bmm(X_pt, WT)
        Y_pt = Y_pt + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor in shape (B, M, K)
        b : Tensor
            Tensor in shape (B, N, K)
        c : Tensor
            Tensor in shape (B, M, N)

        Returns
        -------
        Tensor
            Tensor in shape (B, M, N)
    """

    _SpecializedBase = bmm_rcr

    def __init__(self):
        """Constructor for bmm_rcr_add"""
        super().__init__("r", "c", "r")


class bmm_ccr_add(bmm_xxx_add):
    """Batch GEMM specialization for A[ColMajor], B[ColMajor], C[RowMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()
        D_pt = torch.randn(B, M, N).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        WT = torch.transpose(W_pt, 2, 1)
        Y_pt = torch.bmm(XT, WT)
        Y_pt = Y_pt + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor in shape (B, K, M)
        b : Tensor
            Tensor in shape (B, N, K)
        c : Tensor
            Tensor in shape (B, M, N)

        Returns
        -------
        Tensor
            Tensor in shape (B, M, N)
    """

    _SpecializedBase = bmm_ccr

    def __init__(self):
        """Constructor for bmm_ccr_add"""
        super().__init__("c", "c", "r")


class bmm_rrr_add(bmm_xxx_add):
    """Batch GEMM specialization for A[RowMajor], B[RowMajor], C[RowMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()
        D_pt = torch.randn(B, M, N).cuda().half()

        Y_pt = torch.bmm(X_pt, W_pt) + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor with shape (B, M, K)
        b : Tensor
            Tensor with shape (B, K, N)
        c : Tensor
            Tensor with shape (B, M, N)

        Returns
        -------
        Tensor
            Tensor with shape (B, M, N)
    """

    _SpecializedBase = bmm_rrr

    def __init__(self):
        super().__init__("r", "r", "r")


class bmm_crc_add(bmm_xxx_add):
    """Batch GEMM specialization for A[ColMajor], B[RowMajor], C[ColMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()
        D_pt = torch.randn(B, N, M).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        YT = torch.bmm(XT, W_pt)
        Y_pt = YT.transpose(2, 1) + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor in shape (B, K, M)
        b : Tensor
            Tensor in shape (B, K, N)
        c : Tensor
            Tensor in shape (B, N, M)

        Returns
        -------
        Tensor
            Tensor in shape (B, N, M)

    """

    _SpecializedBase = bmm_crc

    def __init__(self):
        """Constructor for bmm_crc_add"""
        super().__init__("c", "r", "c")


class bmm_rcc_add(bmm_xxx_add):
    """Batch GEMM specialization for A[RowMajor], B[ColMajor], C[ColMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()
        D_pt = torch.randn(B, N, M).cuda().half()

        WT = torch.transpose(W_pt, 2, 1)
        YT = torch.bmm(X_pt, WT)
        Y_pt = YT.transpose(2, 1) + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor in shape (B, M, K)
        b : Tensor
            Tensor in shape (B, N, K)
        c : Tensor
            Tensor in shape (B, N, M)

        Returns
        -------
        Tensor
            Tensor in shape (B, N, M)

    """

    _SpecializedBase = bmm_rcc

    def __init__(self):
        """Constructor for bmm_rcc_add"""
        super().__init__("r", "c", "c")


class bmm_ccc_add(bmm_xxx_add):
    """Batch GEMM specialization for A[ColMajor], B[ColMajor], C[ColMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, K, M).cuda().half()
        W_pt = torch.randn(B, N, K).cuda().half()
        D_pt = torch.randn(B, N, M).cuda().half()

        XT = torch.transpose(X_pt, 2, 1)
        WT = torch.transpose(W_pt, 2, 1)
        YT = torch.bmm(XT, WT)
        Y_pt = YT.transpose(2, 1) + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor in shape (B, K, M)
        b : Tensor
            Tensor in shape (B, N, K)
        c : Tensor
            Tensor in shape (B, N, M)

        Returns
        -------
        Tensor
            Tensor in shape (B, N, M)

    """

    _SpecializedBase = bmm_ccc

    def __init__(self):
        """Constructor for bmm_ccc_add"""
        super().__init__("c", "c", "c")


class bmm_rrc_add(bmm_xxx_add):
    """Batch GEMM specialization for A[RowMajor], B[RowMajor], C[ColMajor] with Add.
    C can be the same size as the output or be broadcast as bias.

    This operator is equivalent to the following PyTorch code:

    .. highlight:: python
    .. code-block:: python

        X_pt = torch.randn(B, M, K).cuda().half()
        W_pt = torch.randn(B, K, N).cuda().half()
        D_pt = torch.randn(B, N, M).cuda().half()
        YT = torch.bmm(X_pt, W_pt)
        Y_pt = YT.transpose(2, 1) + D_pt

    __call__(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        Parameters
        ----------
        a : Tensor
            Tensor with shape (B, M, K)
        b : Tensor
            Tensor with shape (B, K, N)
        c : Tensor
            Tensor with shape (B, N, M)

        Returns
        -------
        Tensor
            Tensor with shape (B, N, M)
    """

    _SpecializedBase = bmm_rrc

    def __init__(self):
        super().__init__("r", "r", "c")
