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

import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor
from aitemplate.utils.torch_utils import string_to_torch_dtype


def get_ait_inputs(
    batch_size=1,
    seq_len=512,
    dtype="int64",
):
    input_ids = Tensor(
        shape=[batch_size, seq_len],
        name="input_ids",
        dtype=dtype,
        is_input=True,
    )
    token_type_ids = Tensor(
        shape=[batch_size, seq_len],
        name="token_type_ids",
        dtype=dtype,
        is_input=True,
    )
    position_ids = Tensor(
        shape=[batch_size, seq_len],
        name="position_ids",
        dtype=dtype,
        is_input=True,
    )
    return (input_ids, token_type_ids, position_ids)


def get_ait_params(
    hidden_size,
    vocab_size,
    max_position_embeddings,
    type_vocab_size,
    dtype="float16",
):
    word_embeddings = Tensor(
        shape=[vocab_size, hidden_size],
        dtype=dtype,
        name="word_embeddings",
        is_input=True,
    )
    token_type_embeddings = Tensor(
        shape=[type_vocab_size, hidden_size],
        dtype=dtype,
        name="token_type_embeddings",
        is_input=True,
    )
    position_embeddings = Tensor(
        shape=[max_position_embeddings, hidden_size],
        dtype=dtype,
        name="position_embeddings",
        is_input=True,
    )
    gamma = Tensor(
        shape=[hidden_size],
        dtype=dtype,
        name="gamma",
        is_input=True,
    )
    beta = Tensor(
        shape=[hidden_size],
        dtype=dtype,
        name="beta",
        is_input=True,
    )
    return (word_embeddings, token_type_embeddings, position_embeddings, gamma, beta)


class bertEmbeddingsTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(bertEmbeddingsTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_bert_embeddings(
        self,
        batch_size,
        seq_len,
        hidden_size,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        test_name="bert_embeddings",
        indices_type="int64",
        input_type="float16",
    ):
        inputs = get_ait_inputs(
            batch_size,
            seq_len,
            dtype=indices_type,
        )
        params = get_ait_params(
            hidden_size,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            dtype=input_type,
        )
        y = ops.bert_embeddings()(*(inputs + params), 1e-5)
        y._attrs["is_output"] = True
        y._attrs["name"] = "output"

        target = detect_target()
        with compile_model(
            y,
            target,
            "./tmp",
            f"{test_name}_{self._test_id}",
        ) as module:
            self._test_id += 1
            torch_indices_type = string_to_torch_dtype(indices_type)
            input_ids = torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                dtype=torch_indices_type,
            ).cuda()
            token_type_ids = torch.randint(
                0,
                type_vocab_size,
                input_ids.size(),
                dtype=torch_indices_type,
            ).cuda()
            position_ids = (
                torch.arange(seq_len, dtype=torch_indices_type)
                .reshape((1, -1))
                .expand(batch_size, -1)
                .contiguous()
                .cuda()
            )
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
            }
            for param in params:
                name = param._attrs["name"]
                shape = [shape.value() for shape in param.shape()]
                w = get_random_torch_tensor(shape, dtype=input_type)
                inputs[name] = w

            word_embedding = torch.nn.functional.embedding(
                input_ids, inputs["word_embeddings"]
            )
            token_type_embedding = torch.nn.functional.embedding(
                token_type_ids, inputs["token_type_embeddings"]
            )
            position_embedding = torch.nn.functional.embedding(
                position_ids, inputs["position_embeddings"]
            )

            pt_embedding = word_embedding + token_type_embedding + position_embedding
            pt_embedding = torch.nn.functional.layer_norm(
                pt_embedding, [hidden_size], inputs["gamma"], inputs["beta"], eps=1e-5
            )

            embedding = torch.empty_like(pt_embedding)
            module.run_with_tensors(inputs, [embedding])
            self.assertTrue(
                torch.allclose(embedding, pt_embedding, atol=1e-2, rtol=1e-2)
            )

    def test_bert_embeddings_fp16(self):
        if detect_target().name() != "rocm":
            self._test_bert_embeddings(
                batch_size=15,
                seq_len=17,
                hidden_size=264,
                vocab_size=10000,
                max_position_embeddings=512,
                type_vocab_size=2,
                test_name="bert_embeddings_fp16",
                input_type="float16",
            )
            self._test_bert_embeddings(
                batch_size=1,
                seq_len=13,
                hidden_size=264,
                vocab_size=10000,
                max_position_embeddings=512,
                type_vocab_size=2,
                test_name="bert_embeddings_fp16",
                input_type="float16",
            )
        self._test_bert_embeddings(
            batch_size=8,
            seq_len=512,
            hidden_size=512,
            vocab_size=10000,
            max_position_embeddings=512,
            type_vocab_size=2,
            test_name="bert_embeddings_fp16",
            input_type="float16",
        )

    def test_bert_embeddings_fp32(self):
        if detect_target().name() != "rocm":
            self._test_bert_embeddings(
                batch_size=15,
                seq_len=17,
                hidden_size=264,
                vocab_size=10000,
                max_position_embeddings=512,
                type_vocab_size=2,
                test_name="bert_embeddings_fp32",
                input_type="float32",
            )
            self._test_bert_embeddings(
                batch_size=1,
                seq_len=13,
                hidden_size=264,
                vocab_size=10000,
                max_position_embeddings=512,
                type_vocab_size=2,
                test_name="bert_embeddings_fp32",
                input_type="float32",
            )
        self._test_bert_embeddings(
            batch_size=8,
            seq_len=512,
            hidden_size=512,
            vocab_size=10000,
            max_position_embeddings=512,
            type_vocab_size=2,
            test_name="bert_embeddings_fp32",
            input_type="float32",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
