#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Neural Chat Chatbot API."""

from .config import NeuralChatConfig, FinetuningConfig, OptimizationConfig
from .pipeline.finetuning.finetuning import Finetuning

class NeuralChatBot:
    def __init__(self, config: NeuralChatConfig):
        self.config = config
        # Initialize other attributes here

    def build_chatbot(self):
        # Implement the logic to build the chatbot
        pass

    def finetune_model(self, config: FinetuningConfig=None):
        # Implement the logic to finetune the model
        config = config if config is not None else self.config.finetune_config
        assert config is not None, "FinetuningConfig is needed for finetuning."
        self.finetuning = Finetuning(config)
        self.finetuning.finetune()

    def optimize_model(self, config: OptimizationConfig):
        # Implement the logic to optimize the model
        pass

    def register_model(self, model):
        # Implement the logic to register the model
        pass

    def predict(self, prompt, temperature, top_p, top_k, repetition_penalty,
                max_new_tokens, do_sample, num_beams, num_return_sequences,
                bad_words_ids, force_words_ids, use_hpu_graphs, use_cache):
        # Implement the logic to generate predictions
        pass

    def predict_stream(self, prompt, temperature, top_p, top_k, repetition_penalty,
                       max_new_tokens, do_sample, num_beams, num_return_sequences,
                       bad_words_ids, force_words_ids, use_hpu_graphs, use_cache):
        # Implement the logic to generate predictions in a streaming manner
        pass
