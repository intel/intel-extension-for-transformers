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
import torch
model_maps = {"gpt_neox": "gptneox", "gpt_bigcode": "starcoder"}

class Model:
    def __init__(self):
        self.module = None
        self.model = None
        self.model_type = None
        self.bin_file = None
        self.generate_round = 0

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
        elif model_name == "gpt_bigcode" or model_name == "starcoder":
            import intel_extension_for_transformers.llm.runtime.graph.starcoder_cpp as cpp_model
        elif model_name == "opt":
            import intel_extension_for_transformers.llm.runtime.graph.opt_cpp as cpp_model
        elif model_name == "bloom":
            import intel_extension_for_transformers.llm.runtime.graph.bloom_cpp as cpp_model
        elif model_name == "chatglm":
            import intel_extension_for_transformers.llm.runtime.graph.chatglm_cpp as cpp_model
        elif model_name == "chatglm2":
            import intel_extension_for_transformers.llm.runtime.graph.chatglm2_cpp as cpp_model
        elif model_name == "baichuan":
            import intel_extension_for_transformers.llm.runtime.graph.baichuan_cpp as cpp_model
        elif model_name == "polyglot":
            import intel_extension_for_transformers.llm.runtime.graph.polyglot_cpp as cpp_model
        else:
            raise TypeError("Unspported model type {}!".format(model_name))
        self.module = cpp_model

    def init(self, model_name, **kwargs):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = model_maps.get(config.model_type, config.model_type)
        if model_type == "chatglm" and "chatglm2" in config._name_or_path:
            model_type = "chatglm2"
        self.__import_package(model_type)

        # 1. convert model
        fp32_bin = "ne_{}_f32.bin".format(model_type)
        convert_model(model_name, fp32_bin, "f32")
        assert os.path.exists(fp32_bin), "Fail to convert pytorch model"

        # 2. quant model
        quant_bin = "ne_{}_q.bin".format(model_type)
        self.module.Model.quant_model(model_path = fp32_bin, out_path = quant_bin, **kwargs)
        assert os.path.exists(quant_bin), "Fail to quantize model"
        
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


    def generate(self, input_ids, streamer=None, interactive=False, ignore_prompt=False, **kwargs):
        if self.model is None:
            self.init_from_bin(self.model_type, self.bin_file, **kwargs)
            self.generate_round = 0
        elif not interactive:
            self.model.reinit()
            self.generate_round = 0

        ret = [[]]
        if self.generate_round == 0 and not ignore_prompt:
            ret = input_ids.tolist()

        # TODO support multi batch
        assert input_ids.shape[0] == 1, "Unsupport multi-batch input ids."
        beam_search = False
        if ("num_beams" in kwargs and kwargs["num_beams"] > 1) and not \
            kwargs.get("do_sample", False):
            beam_search = True
        if streamer:
            if beam_search:
                print("ERROR, can not use streamer when use beam search for generation!")
                import sys
                sys.exit(1)
            if self.generate_round == 0 and not ignore_prompt:
                streamer.put(input_ids)
            if interactive:
                self.model.reset_token_end()
            while not self.is_token_end():
                out = self.model.generate(input_ids = input_ids.tolist()[0])
                if len(out) == 0:
                    break
                streamer.put(torch.tensor([out]))
                ret[0].extend(out)
            streamer.end()
        else:
            ret[0].extend(self.model.generate_tokens(input_ids = input_ids.tolist()[0]))
        
        self.generate_round += 1
        return ret

    def is_token_end(self):
        return self.model.is_token_end()
