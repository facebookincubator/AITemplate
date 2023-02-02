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
import torch
from transformers import AutoModelForMaskedLM, BertForMaskedLM


class BertBaseUncased:
    def __init__(self, model_path="bert-base-uncased", pretrained=True):
        if not pretrained:
            pretrained = AutoModelForMaskedLM.from_pretrained(model_path)
            self._model = BertForMaskedLM(pretrained.config).cuda().half()
        else:
            self._model = AutoModelForMaskedLM.from_pretrained(model_path).cuda().half()
        self._vocab_size = 30522

    def forward(self, *args, **kwargs):
        # runs the full model with classification head
        outputs = self._model(*args, **kwargs)
        return outputs.logits

    def generate_inputs(self, batch_size, seq_len):
        dtype = torch.long
        input_ids = torch.randint(
            0, self._vocab_size, (batch_size, seq_len), dtype=dtype
        ).cuda()
        token_type_ids = torch.zeros(input_ids.size(), dtype=dtype).cuda()
        position_ids = (
            torch.arange(seq_len, dtype=dtype)
            .reshape((1, -1))
            .expand(batch_size, -1)
            .contiguous()
            .cuda()
        )
        return (input_ids, token_type_ids, position_ids)

    def get_parameters(self):
        return dict(self._model.named_parameters())
