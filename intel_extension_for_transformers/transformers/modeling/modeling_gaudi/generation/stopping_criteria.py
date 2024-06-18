# coding=utf-8
# Copyright 2022 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from optimum.utils import logging


logger = logging.get_logger(__name__)


def gaudi_MaxLengthCriteria_call(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    token_idx = kwargs.get("token_idx", None)
    if token_idx is not None:
        return token_idx >= self.max_length
    else:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return is_done


def gaudi_MaxNewTokensCriteria_call(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    token_idx = kwargs.get("token_idx", None)
    if token_idx is not None:
        return token_idx >= self.max_length
    else:
        return input_ids.shape[-1] >= self.max_length
