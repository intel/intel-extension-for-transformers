# !/usr/bin/env python
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

import time
import spacy
from transformers import AutoTokenizer, TextIteratorStreamer
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM, 
    WeightOnlyQuantConfig
)
from .utils.utils import (
    enforce_stop_tokens, 
    get_current_time
)
from .utils.process_text import process_time, process_entities
from intel_extension_for_transformers.neural_chat.prompts import PromptTemplate


class NamedEntityRecognitionINT():
    """
        Initialize spacy model and llm model
        If you want to inference with int8 model, set compute_dtype='fp32' and weight_dtype='int8';
        If you want to inference with int4 model, set compute_dtype='int8' and weight_dtype='int4'.
    """

    def __init__(self, 
                 model_path="/home/tme/Llama-2-7b-chat-hf/", 
                 spacy_model="en_core_web_lg", 
                 compute_dtype="fp32", 
                 weight_dtype="int8",
                 device="cpu") -> None:
        self.nlp = spacy.load(spacy_model)
        config = WeightOnlyQuantConfig(compute_dtype=compute_dtype, weight_dtype=weight_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          quantization_config=config, 
                                                          trust_remote_code=True)
        print("[NER info] Spacy and LLM model initialized.")


    def inference(self, query: str, prompt: str=None, threads: int=52, max_new_tokens: int=32, seed: int=1234):
        start_time = time.time()
        cur_time = get_current_time()
        print("[NER info] Current time is:{}".format(cur_time))
        if not prompt:
            pt = PromptTemplate("ner")
            pt.append_message(pt.conv.roles[0], cur_time)
            pt.append_message(pt.conv.roles[1], query)
            prompt = pt.get_prompt()
        print("[NER info] Prompt is: ", prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        streamer = TextIteratorStreamer(self.tokenizer)

        result_tokens = self.model.generate(
            inputs, 
            streamer=streamer, 
            max_new_tokens=max_new_tokens, 
            seed=seed, 
            threads=threads
        )
        self.model.model.reinit()
        gen_text = self.tokenizer.batch_decode(result_tokens)

        result_text = enforce_stop_tokens(gen_text[0])
        doc = self.nlp(result_text)
        mentioned_time = process_time(result_text, doc)

        new_doc = self.nlp(query)
        result = process_entities(query, new_doc, mentioned_time)
        print("[NER info] Inference time consumption: ", time.time() - start_time)

        return result
