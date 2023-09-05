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
from transformers import AutoConfig
from intel_extension_for_transformers.llm.runtime.graph.scripts.convert_model import convert_model

model_maps = {"gpt_neox": "gptneox", "RefinedWebModel": "falcon"}

class Model:
    def __init__(self):
        self.module = None
        self.model = None

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

    def init(self, model_name,
             bits = 4, alg = "sym", block_size = 32, scale_dtype = "fp32", compute_type = "ggml",
             n_predict = -1, batch_size = 512, ctx_size = 512, seed = -1, threads = 8, repeat_penalty = 1.1):
        config = AutoConfig.from_pretrained(model_name)
        model_type = model_maps.get(config.model_type, config.model_type)
        self.__import_package(model_type)

        # 1. convert model
        convert_model(model_name, "ne_{}_f32.bin".format(model_type), "f32")

        # 2. quant model
        quant_bin = "ne_{}_q{}_{}_{}_{}_{}.bin".format(model_type, bits, alg, block_size, scale_dtype, compute_type)
        self.module.Model.quant_model(model_path = "ne_{}_f32.bin".format(model_type),
                                    out_path = quant_bin,
                                    bits = bits,
                                    alg = alg,
                                    block_size = block_size,
                                    scale_dtype = scale_dtype,
                                    compute_type = compute_type)
        
        self.init_from_bin(model_type, quant_bin,
                            n_predict = n_predict,
                            batch_size = batch_size,
                            ctx_size = ctx_size,
                            seed = seed, 
                            threads = threads,
                            repeat_penalty = repeat_penalty
                            )
        
        # clean 

    def init_from_bin(self, model_name, model_path,
                      n_predict = -1, batch_size = 512, ctx_size = 512, seed = -1, threads = 8, repeat_penalty = 1.1):
        self.__import_package(model_name)
        self.model = self.module.Model()
        self.model.init_model(model_path,
                              n_predict = n_predict,
                              batch_size = batch_size,
                              ctx_size = ctx_size,
                              seed = seed, 
                              threads = threads,
                              repeat_penalty = repeat_penalty
                              )

    def quant_model(self, model_name, model_path, out_path,
                    bits = 4, alg = "sym", block_size = 32, scale_dtype = "fp32", compute_type = "ggml"):
        self.__import_package(model_name)
        self.module.Model.quant_model(model_path = model_path,
                                    out_path = out_path,
                                    bits = bits,
                                    alg = alg,
                                    block_size = block_size,
                                    scale_dtype = scale_dtype,
                                    compute_type = compute_type)

    def generate(self, prompt, stream_mode = True):
        out = self.model.generate(prompt = prompt, stream_mode = stream_mode)
        return out

    def is_token_end(self):
        return self.model.is_token_end()
