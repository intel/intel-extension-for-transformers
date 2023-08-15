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

from .base_model import BaseModel, register_model_adapter
import logging
from fastchat.conversation import get_conv_template, Conversation
from neural_chat.pipeline.inference.inference import load_model, predict, predict_stream

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class LlamaModel(BaseModel):
    def load_model(self, kwargs: dict):
        print("Loading model {}".format(kwargs["model_name"]))
        load_model(model_name=kwargs["model_name"],
                   tokenizer_name=kwargs["tokenizer_name"],
                   device=kwargs["device"],
                   use_hpu_graphs=kwargs["use_hpu_graphs"],
                   cpu_jit=kwargs["cpu_jit"],
                   use_cache=kwargs["use_cache"],
                   peft_path=kwargs["peft_path"],
                   use_deepspeed=kwargs["use_deepspeed"])

    def match(self, model_path: str):
        return "llama" in model_path.lower()

    def predict_stream(self, params):
        """
        Generates streaming text based on the given parameters and prompt.

        Args:
            params (dict): A dictionary containing the parameters for text generation.
            `device` (string): Specifies the device type for text generation. It can be either "cpu" or "hpu".
            `prompt` (string): Represents the initial input or context provided to the text generation model.
            `temperature` (float): Controls the randomness of the generated text.
                                Higher values result in more diverse outputs.
            `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                            Smaller values make the output more focused.
            `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
            `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                        Higher values discourage repetition.
            `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
            `do_sample` (bool): Determines whether to use sampling-based text generation.
                                If set to True, the output will be sampled; otherwise,
                                it will be determined by the model's top-k or top-p strategy.
            `num_beams` (int): Controls the number of beams used in beam search.
                            Higher values increase the diversity but also the computation time.
            `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                                If not provided, the default model is "mosaicml/mpt-7b-chat".
            `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
            `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
            `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
            `use_hpu_graphs` (bool): Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.
            `use_cache` (bool): Determines whether to utilize kv cache for accelerated generation.

        Returns:
            generator: A generator that yields the generated streaming text.
        """
        return predict_stream(params)

    def predict(self, params):
        """
        Generates streaming text based on the given parameters and prompt.

        Args:
            params (dict): A dictionary containing the parameters for text generation.
            `prompt` (string): Represents the initial input or context provided to the text generation model.
            `temperature` (float): Controls the randomness of the generated text.
                                Higher values result in more diverse outputs.
            `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                            Smaller values make the output more focused.
            `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
            `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                        Higher values discourage repetition.
            `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
            `do_sample` (bool): Determines whether to use sampling-based text generation.
                                If set to True, the output will be sampled; otherwise,
                                it will be determined by the model's top-k or top-p strategy.
            `num_beams` (int): Controls the number of beams used in beam search.
                            Higher values increase the diversity but also the computation time.
            `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                                If not provided, the default model is "mosaicml/mpt-7b-chat".
            `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
            `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
            `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
            `use_hpu_graphs` (bool): Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.
            `use_cache` (bool): Determines whether to utilize kv cache for accelerated generation.

        Returns:
            output: model generated text.
        """
        return predict(params)

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")

register_model_adapter(LlamaModel)
