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
from typing import Tuple

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target

# pylint: disable=W0102

USE_CUDA = detect_target().name() == "cuda"


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps):
        """dense + add is included in nn.MultiheadAttention.
        This class now only contains LayerNorm.
        """
        super().__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if not USE_CUDA:
            hidden_states = (
                hidden_states
                if hidden_states._rank() == 2
                else ops.reshape()(hidden_states, [-1, hidden_states._size(-1)])
            )
        # [B, S, H] on cuda, [B * S, H] on rocm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        hidden_size,
        num_attention_heads,
        layer_norm_eps,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    ):
        super().__init__()
        self.self = nn.MultiheadAttention(
            dim=hidden_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_attention_heads,
            qkv_bias=True,
            attn_drop=attention_probs_dropout_prob,
            proj_drop=hidden_dropout_prob,
            has_residual=True,
        )
        self.output = BertSelfOutput(hidden_size, layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor]:
        self_output = self.self(hidden_states, hidden_states)
        attention_output = self.output(self_output)
        outputs = (attention_output,)
        return outputs


# FFN block
class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        # dense + activation
        self.dense = nn.Linear(
            hidden_size, intermediate_size, specialization=hidden_act
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, layer_norm_eps, hidden_dropout_prob
    ):
        super().__init__()
        assert hidden_dropout_prob == 0.0
        # dense + add
        self.dense = nn.Linear(intermediate_size, hidden_size, specialization="add")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states, input_tensor)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        batch_size,
        seq_len,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super().__init__()
        self.attention = BertAttention(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            layer_norm_eps=layer_norm_eps,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(
            hidden_size, intermediate_size, layer_norm_eps, hidden_dropout_prob
        )

    def feed_forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(
        self,
        hidden_states: Tensor,
    ):
        # [B, S, H]
        shape = hidden_states.shape()
        # [B, S, H] on cuda, [B * S, H] on rocm
        self_attention_outputs = self.attention(hidden_states)
        layer_output = self.feed_forward(self_attention_outputs[0])
        # [B * S, H] to [B, S, H] on rocm
        layer_output = (
            layer_output
            if layer_output._rank() == 3
            else ops.reshape()(layer_output, shape)
        )
        return (layer_output,)


class BertEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        batch_size,
        seq_len,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                BertLayer(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    hidden_dropout_prob=hidden_dropout_prob,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
    ):
        for layer_module in self.layer:
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

        return layer_outputs


class BertModel(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        num_hidden_layers,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        add_pooling_layer=False,
    ):
        super().__init__()
        assert not add_pooling_layer

        self.embeddings = nn.BertEmbeddings(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.encoder = BertEncoder(
            batch_size=batch_size,
            seq_len=seq_len,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        position_ids: Tensor,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
        )
        return encoder_outputs


class BertModelEncodersOnly(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        num_hidden_layers,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        add_pooling_layer=False,
    ):
        super().__init__()
        assert not add_pooling_layer

        self.encoder = BertEncoder(
            batch_size=batch_size,
            seq_len=seq_len,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
        )

    def forward(
        self,
        encoder_input: Tensor,
    ):
        encoder_outputs = self.encoder(encoder_input)
        return encoder_outputs


class BertBaseUncased(nn.Module):
    """Bert base uncased with no classification head."""

    def __init__(
        self,
        batch_size,
        seq_len,
        vocab_size=30522,
        max_position_embeddings=512,
        type_vocab_size=2,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    ):
        super().__init__()
        self.bert = BertModel(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            add_pooling_layer=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return outputs


class BertBaseEncodersOnly(nn.Module):
    """Bert base uncased with no classification head and no embeddings."""

    def __init__(
        self,
        batch_size,
        seq_len,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    ):
        super().__init__()
        self.bert = BertModelEncodersOnly(
            batch_size=batch_size,
            seq_len=seq_len,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            add_pooling_layer=False,
        )

    def forward(
        self,
        encoder_input: Tensor,
    ) -> Tensor:
        outputs = self.bert(encoder_input)
        return outputs
