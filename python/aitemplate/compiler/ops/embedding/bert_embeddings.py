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
Operator definition for bert_embeddings.
"""
from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm, Operator, Tensor
from aitemplate.utils import shape_utils


class bert_embeddings(Operator):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "bert_embeddings"
        self._attrs["has_profiler"] = False

    def _infer_shapes(self, x, embeddings):
        return x.shape() + [embeddings._size(-1)]

    def __call__(
        self,
        input_ids,  # [B, S] or [B * S]
        token_type_ids,  # [B, S] or [B * S]
        position_ids,  # [B, S] or [B * S]
        word_embeddings,  # [vocab_size, hidden_size]
        token_type_embeddings,  # [type_vocab_size, hidden_size]
        position_embeddings,  # [max_position_embeddings, hidden_size]
        gamma,  # [hidden_size]
        beta,  # [hidden_size]
        eps=1e-5,
    ) -> Tensor:
        """Compute
        embedding = layernorm(word_embedding + token_type_embedding + position_embedding).

        Parameters
        ----------
        input_ids, token_type_ids, position_ids must have the same sizes, either 2D or 1D.

        Returns
        -------
        The computed embedding.
        """

        # dtype check
        dtype_input_ids = input_ids._attrs["dtype"]
        dtype_token_type_ids = token_type_ids._attrs["dtype"]
        dtype_position_ids = position_ids._attrs["dtype"]
        assert (
            dtype_input_ids == dtype_token_type_ids
            and dtype_input_ids == dtype_position_ids
        ), "dtype of input_ids, token_type_ids, and position_ids must be the same"

        dtype_word_embeddings = word_embeddings._attrs["dtype"]
        dtype_token_type_embeddings = token_type_embeddings._attrs["dtype"]
        dtype_position_embeddings = position_embeddings._attrs["dtype"]
        assert (
            dtype_word_embeddings == dtype_token_type_embeddings
            and dtype_word_embeddings == dtype_position_embeddings
        ), "dtype of word_embeddings, token_type_embeddings, position_embeddings must be the same"

        assert dtype_input_ids in [
            "int",
            "int32",
            "int64",
        ], f"Expected dtype int/int32/int64 for index, got dtype {dtype_input_ids}"

        assert dtype_word_embeddings in [
            "float16",
            "float32",
        ], f"Expected dtype float16/float32 for embeddings, got dtype {dtype_word_embeddings}"

        # expecting all three ids to have the same shapes
        assert shape_utils.is_same_shape(input_ids.shape(), token_type_ids.shape()), (
            f"Expecting input_ids and token_type_ids to have the same shapes, but got "
            f"input_ids.shape(): {input_ids.shape()}, token_type_ids.shape(): {token_type_ids.shape()}"
        )
        assert shape_utils.is_same_shape(input_ids.shape(), position_ids.shape()), (
            f"Expecting input_ids and position_ids to have the same shapes, but got "
            f"input_ids.shape(): {input_ids.shape()}, position_ids.shape(): {position_ids.shape()}"
        )

        # expecting the last dim of all three embedding tables to be the same
        dim = word_embeddings._size(-1)
        assert isinstance(dim, IntImm), f"Embedding dim {dim} must be static."
        dim_value = dim.value()
        assert dim_value % 8 == 0, f"Embedding dim {dim} must be multiple of 8."
        assert dim == token_type_embeddings._size(-1), (
            f"Expecting the last dim of word_embeddings and token_type_embeddings to be the same, "
            f"but got {word_embeddings._size(-1)} and {token_type_embeddings._size(-1)}"
        )
        assert dim == position_embeddings._size(-1), (
            f"Expecting the last dim of word_embeddings and position_embeddings to be the same, "
            f"but got {word_embeddings._size(-1)} and {position_embeddings._size(-1)}"
        )

        self._attrs["eps"] = eps

        self._attrs["inputs"] = [
            input_ids,
            token_type_ids,
            position_ids,
            word_embeddings,
            token_type_embeddings,
            position_embeddings,
            gamma,
            beta,
        ]

        self._set_depth()

        output_shape = self._infer_shapes(input_ids, word_embeddings)
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=word_embeddings._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(self._attrs)
