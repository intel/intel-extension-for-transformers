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

import torch
from intel_extension_for_transformers.llm.runtime.graph.scripts.convert import convert_model
from transformers import AutoConfig, AutoTokenizer

model_maps = {"gpt_neox": "gptneox", "gpt_bigcode": "starcoder"}


class Model:
    def __init__(self):
        self.module = None
        self.model = None
        self.model_type = None
        self.bin_file = None
        self.generate_round = 0

    def __import_package(self, model_type):
        if self.module:
            return
        if model_type == "gptj":
            import intel_extension_for_transformers.llm.runtime.graph.gptj_cpp as cpp_model
        elif model_type == "falcon":
            import intel_extension_for_transformers.llm.runtime.graph.falcon_cpp as cpp_model
        elif model_type == "gptneox":
            import intel_extension_for_transformers.llm.runtime.graph.gptneox_cpp as cpp_model
        elif model_type == "dolly":
            import intel_extension_for_transformers.llm.runtime.graph.dolly_cpp as cpp_model
        elif model_type == "llama" or model_type == "llama2":
            import intel_extension_for_transformers.llm.runtime.graph.llama_cpp as cpp_model
        elif model_type == "mpt":
            import intel_extension_for_transformers.llm.runtime.graph.mpt_cpp as cpp_model
        elif model_type == "gpt_bigcode" or model_type == "starcoder":
            import intel_extension_for_transformers.llm.runtime.graph.starcoder_cpp as cpp_model
        elif model_type == "opt":
            import intel_extension_for_transformers.llm.runtime.graph.opt_cpp as cpp_model
        elif model_type == "bloom":
            import intel_extension_for_transformers.llm.runtime.graph.bloom_cpp as cpp_model
        elif model_type == "chatglm":
            import intel_extension_for_transformers.llm.runtime.graph.chatglm_cpp as cpp_model
        elif model_type == "chatglm2":
            import intel_extension_for_transformers.llm.runtime.graph.chatglm2_cpp as cpp_model
        elif model_type == "baichuan":
            import intel_extension_for_transformers.llm.runtime.graph.baichuan_cpp as cpp_model
        elif model_type == "polyglot":
            import intel_extension_for_transformers.llm.runtime.graph.polyglot_cpp as cpp_model
        elif model_type == "mistral":
            import intel_extension_for_transformers.llm.runtime.graph.mistral_cpp as cpp_model
        else:
            raise TypeError("Unspported model type {}!".format(model_type))
        self.module = cpp_model

    @staticmethod
    def get_model_type(model_config):
        model_type = model_maps.get(model_config.model_type, model_config.model_type)
        if model_type == "chatglm" and "chatglm2" in model_config._name_or_path:
            model_type = "chatglm2"
        return model_type

    def init(self, model_name, not_quant=False, use_cache=False, **quant_kwargs):
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_type = Model.get_model_type(self.config)
        self.__import_package(model_type)

        # check cache and quantization
        output_path = "runtime_outs"
        os.makedirs(output_path, exist_ok=True)
        fp32_bin = "{}/ne_{}_f32.bin".format(output_path, model_type)
        quant_desc = quant_kwargs['weight_dtype']
        if quant_kwargs['use_ggml']:
            quant_desc += "_ggml"
        else:
            quant_desc += "_jblas_c" + quant_kwargs['compute_dtype']
            if quant_kwargs['group_size'] == -1:
                quant_desc += "_pc"
            else:
                quant_desc += "_g{}".format(quant_kwargs['group_size'])
        quant_bin = "{}/ne_{}_q_{}.bin".format(output_path, model_type, quant_desc)

        if not_quant:
            self.bin_file = fp32_bin
        else:
            self.bin_file = quant_bin
        if use_cache and os.path.exists(self.bin_file):
            return

        if not use_cache or not os.path.exists(fp32_bin):
            convert_model(model_name, fp32_bin, "f32")
            assert os.path.exists(fp32_bin), "Fail to convert pytorch model"

        if not_quant:
            print("FP32 model will be used.")
            return
        self.module.Model.quant_model(model_path=fp32_bin, out_path=quant_bin, **quant_kwargs)
        assert os.path.exists(quant_bin), "Fail to quantize model"

        # clean
        if not use_cache:
            os.remove(fp32_bin)

    def init_from_bin(self, model_type, model_path, **generate_kwargs):
        self.__import_package(model_type)
        self.model = self.module.Model()
        if "threads" not in generate_kwargs:
            threads = os.getenv("OMP_NUM_THREADS")
            if threads is None:
                generate_kwargs["threads"] = len(os.sched_getaffinity(0))
            else:
                generate_kwargs["threads"] = int(threads)
        self.model.init_model(model_path, **generate_kwargs)

    def quant_model(self, model_type, model_path, out_path, **quant_kwargs):
        self.__import_package(model_type)
        self.module.Model.quant_model(model_path=model_path, out_path=out_path, **quant_kwargs)

    def generate(self, input_ids, streamer=None, interactive=False, ignore_prompt=False, stopping_criteria=None,  **generate_kwargs):
        max_new_tokens = generate_kwargs.get("max_new_tokens", -1)
        if self.model is None:
            self.init_from_bin(self.model_type, self.bin_file, batch_size=input_ids.shape[0],
                               **generate_kwargs)
            self.generate_round = 0
        elif not interactive:
            self.model.reinit()
            self.generate_round = 0

        ret = [[]]
        if self.generate_round == 0 and not ignore_prompt:
            ret = input_ids.tolist()

        beam_search = False
        if (generate_kwargs.get("num_beams", 1) > 1) and not generate_kwargs.get("do_sample", False):
            beam_search = True
        if not beam_search:
            # TODO support multi batch
            assert input_ids.shape[0] == 1, "Unsupport multi-batch input ids."

        if streamer:
            assert input_ids.shape[0] == 1, "Streamer only supports batch size 1."
            assert beam_search == False, "ERROR, can not use streamer when use beam search for generation! \
                Make sure that `num_beams` is set to 1."
            if self.generate_round == 0 and not ignore_prompt:
                streamer.put(input_ids)

        if interactive:
            self.model.reset_token_end()
        out_count = 0
        input_list = input_ids.tolist()
        while True:
            response = self.model.generate(input_ids=input_list)
            input_list = []  # next-token stage will use previous output
            if len(response) == 0:
                break
            if streamer:
                streamer.put(torch.tensor([response[0]]))
            for i in range(len(response)):
                ret[i].extend(response[i])
            if beam_search:
                break
            if stopping_criteria is not None:
                if stopping_criteria(torch.tensor(ret), None):
                    break
            elif ret[0][-1] == self.tokenizer.eos_token_id or \
                    (max_new_tokens != -1 and out_count > max_new_tokens):
                break
            out_count += 1
        if streamer:
            streamer.end()

        self.generate_round += 1
        return ret

    def is_token_end(self):
        return self.model.is_token_end()

    def __call__(self, input_ids, reinit=False, **kwargs):
        if self.model is None:
            self.init_from_bin(self.model_type, self.bin_file, **kwargs)
            self.generate_round = 0
        elif reinit:
            self.model.reinit()
            self.generate_round = 0
        return self.model.evaluate(input_ids.tolist())
