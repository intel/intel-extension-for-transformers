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
import os
from transformers import AutoConfig
from intel_extension_for_transformers.llm.runtime.graph.scripts.convert import convert_model

model_maps = {"gpt_neox": "gptneox", "RefinedWebModel": "falcon"}

class Model:
    def __init__(self):
        self.module = None
        self.model = None
        self.model_type = None
        self.bin_file = None

    def __import_package(self, model_name):
        if self.module:
            return
        if model_name == "gptj":
            import intel_extension_for_transformers.llm.runtime.graph.gptj_cpp as cpp_model
        elif model_name == "falcon":
            import intel_extension_for_transformers.llm.runtime.graph.falcon_cpp as cpp_model
        elif model_name == "gptneox":
            import intel_extension_for_transformers.llm.runtime.graph.gptneox_cpp as cpp_model
        elif model_name == "dolly":
            import intel_extension_for_transformers.llm.runtime.graph.dolly_cpp as cpp_model
        elif model_name == "llama" or model_name == "llama2":
            import intel_extension_for_transformers.llm.runtime.graph.llama_cpp as cpp_model
        elif model_name == "mpt":
            import intel_extension_for_transformers.llm.runtime.graph.mpt_cpp as cpp_model
        elif model_name == "starcoder":
            import intel_extension_for_transformers.llm.runtime.graph.starcoder_cpp as cpp_model
        elif model_name == "opt":
            import intel_extension_for_transformers.llm.runtime.graph.opt_cpp as cpp_model
        elif model_name == "bloom":
            import intel_extension_for_transformers.llm.runtime.graph.bloom_cpp as cpp_model
        elif model_name == "chatglm2":
            import intel_extension_for_transformers.llm.runtime.graph.chatglm2_cpp as cpp_model
        else:
            raise TypeError("Unspported model type {}!".format(model_name))
        self.module = cpp_model

    def init(self, model_name, **kwargs):
        config = AutoConfig.from_pretrained(model_name)
        model_type = model_maps.get(config.model_type, config.model_type)
        self.__import_package(model_type)

        # 1. convert model
        fp32_bin = "ne_{}_f32.bin".format(model_type)
        convert_model(model_name, fp32_bin, "f32")

        # 2. quant model
        quant_bin = "ne_{}_q.bin".format(model_type)
        self.module.Model.quant_model(model_path = fp32_bin, out_path = quant_bin, **kwargs)
        
        self.model_type = model_type
        self.bin_file = quant_bin
        
        # clean
        os.remove(fp32_bin)


    def init_from_bin(self, model_name, model_path, **kwargs):
        self.__import_package(model_name)
        self.model = self.module.Model()
        self.model.init_model(model_path, **kwargs)

    def quant_model(self, model_name, model_path, out_path, **kwargs):
        self.__import_package(model_name)
        self.module.Model.quant_model(model_path = model_path,
                                    out_path = out_path, **kwargs)

    def generate(self, prompt, streamer = None, sentence_mode = True, **kwargs):
        # TODO support streamer
        if self.model is None:
            self.init_from_bin(self.model_type, self.bin_file, **kwargs)
        
        out = self.model.generate(prompt = prompt, sentence_mode = sentence_mode)
        return out

    def is_token_end(self):
        return self.model.is_token_end()
