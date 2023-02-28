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

from ...compiler import ops
from .dropout import Dropout
from .layer_norm import LayerNorm
from .module import Module
from .parameter import Parameter


class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        shape (List[int]): denotes the shape of the embeddings which is typically `[num_embeddings, embedding_dim]` where `num_embeddings` is the size of the dictionary of embeddings, and `embedding_dim` is the size of each embedding vector.
        dtype (string): denotes the data type
    """

    def __init__(
        self,
        shape,
        dtype,
    ):
        super().__init__()
        self.weight = Parameter(shape=shape, dtype=dtype)

    def tensor(self):
        return self.weight.tensor()


class BertEmbeddings(Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    USE_CUDA = None

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        layer_norm_eps,
        hidden_dropout_prob,
        dtype="float16",
    ):
        super().__init__()
        if BertEmbeddings.USE_CUDA is None:
            BertEmbeddings.USE_CUDA = detect_target().name() == "cuda"
        assert (
            hidden_dropout_prob == 0.0
        ), "Dropout rate larger than 0 is not supported yet."

        self.word_embeddings = Embedding(shape=[vocab_size, hidden_size], dtype=dtype)
        self.position_embeddings = Embedding(
            shape=[max_position_embeddings, hidden_size],
            dtype=dtype,
        )
        self.token_type_embeddings = Embedding(
            shape=[type_vocab_size, hidden_size], dtype=dtype
        )

        self.LayerNorm = LayerNorm([hidden_size], layer_norm_eps, dtype)
        self.dropout = Dropout(hidden_dropout_prob)

    def forward(
        self,
        input_ids,  # [B, S]
        token_type_ids,  # [B, S]
        position_ids,  # [B, S]
    ):
        embeddings = ops.bert_embeddings()(
            input_ids,
            token_type_ids,
            position_ids,
            self.word_embeddings.weight.tensor(),
            self.token_type_embeddings.weight.tensor(),
            self.position_embeddings.weight.tensor(),
            self.LayerNorm.weight.tensor(),
            self.LayerNorm.bias.tensor(),
            self.LayerNorm.eps,
        )
        embeddings = self.dropout(embeddings)
        return embeddings
