# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
special gemm ops
"""
from .bmm_rcr_n1 import bmm_rcr_n1
from .bmm_rrr_k1_tanh import bmm_rrr_k1_tanh
from .gemm_rrr_small_nk import gemm_rrr_small_nk


__all__ = ["bmm_rcr_n1", "bmm_rrr_k1_tanh", "gemm_rrr_small_nk"]
