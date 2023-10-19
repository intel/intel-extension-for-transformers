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

from threading import Thread
import re
import time
import torch
import spacy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    AutoConfig,
)
import intel_extension_for_pytorch as intel_ipex
from .utils.utils import (
    enforce_stop_tokens,
    get_current_time
)
from .utils.process_text import process_time, process_entities
from intel_extension_for_transformers.neural_chat.prompts import PromptTemplate


class NamedEntityRecognition():
    """
        NER class to inference with fp32 or bf16 llm models
        Set bf16=True if you want to inference with bf16 model.
    """

    def __init__(self, model_path="./Llama-2-7b-chat-hf/", spacy_model="en_core_web_lg", bf16: bool=False) -> None:
        # initialize tokenizer and models
        self.nlp = spacy.load(spacy_model)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.init_device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False if (re.search("llama", model_path, re.IGNORECASE)
                or re.search("neural-chat-7b-v2", model_path, re.IGNORECASE)) else True,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            config=config,
            device_map="auto",
            trust_remote_code=True
        )
        self.bf16 = bf16
        # optimize model with ipex if bf16
        if bf16:
            self.model = intel_ipex.optimize(
                self.model.eval(),
                dtype=torch.bfloat16,
                inplace=True,
                level="O1",
                auto_kernel_selection=True,
            )
            for i in range(3):
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id = i
        print("[NER info] Spacy and LLM model initialized.")


    def inference(self, 
                  query: str, 
                  prompt: str=None, 
                  max_new_tokens: int=32, 
                  temperature: float=0.01, 
                  top_k: int=3, 
                  repetition_penalty: float=1.1):
        start_time = time.time()
        cur_time = get_current_time()
        print("[NER info] Current time is:{}".format(cur_time))
        if not prompt:
            pt = PromptTemplate("ner")
            pt.append_message(pt.conv.roles[0], cur_time)
            pt.append_message(pt.conv.roles[1], query)
            prompt = pt.get_prompt()
        print("[NER info] Prompt is: ", prompt)
        inputs= self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        
        # define generate kwargs and construct inferencing thread
        if self.bf16:
            generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            def generate_output():
                dtype = self.model.dtype if hasattr(self.model, 'dtype') else torch.bfloat16
                try:
                    with torch.no_grad():
                        context = torch.cpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=True)
                        with context:
                            output_token=self.model.generate(
                                **inputs,
                                **generate_kwargs,
                                streamer=streamer,
                                return_dict_in_generate=True,
                            )
                            return output_token
                except Exception as e:
                    raise Exception(e)

            thread = Thread(target=generate_output)

        else:
            generate_kwargs = dict(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                streamer=streamer,
            )
            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)

        # inference with thread
        thread.start()
        text = ""
        for new_text in streamer:
            text += new_text
            
        # process inferenced tokens and texts
        text = enforce_stop_tokens(text)
        doc = self.nlp(text)
        mentioned_time = process_time(text, doc)

        new_doc = self.nlp(query)
        result = process_entities(query, new_doc, mentioned_time)
        print("[NER info] Inference time consumption: ", time.time() - start_time)

        return result

