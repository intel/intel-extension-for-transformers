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

from .base_model import BaseModel
from fastchat.conversation import get_conv_template, Conversation
from datetime import datetime
import torch
import logging
from typing import List
from transformers import (
    AutoModel,
    AutoTokenizer,
    StoppingCriteriaList,
)
from .utils import set_cpu_running_env, import_deepspeed, smart_context_manager, init_deepspeed_inference
from .utils import InvalidScoreLogitsProcessor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


invalid_score_processor = InvalidScoreLogitsProcessor()

class ChatGlmModel(BaseModel):
    def init_model(self, config):
        print("Loading model {}".format(config.model_name_or_path))
        self.config = config
        if config.device == "cpu":
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                logger.warn(
                    "Intel Extension for PyTorch is not installed, but is required for xpu inference."
                )
            set_cpu_running_env()
            torch_dtype = torch.bfloat16
        elif config.device == "cuda":
            torch_dtype = torch.float16
        elif config.device == "hpu":
            if config.use_deepspeed:
                import_deepspeed()
            # Tweak generation so that it runs faster on Gaudi
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()

        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        with smart_context_manager(use_deepspeed=config.use_deepspeed):
            if config.device == "cuda":
                model = AutoModel.from_pretrained(config.model_name_or_path, trust_remote_code=True).cuda().half()
            else:
                model = AutoModel.from_pretrained(
                    config.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        model = model.eval()
        if (
            hasattr(model.generation_config, "pad_token_id")
            and model.generation_config.pad_token_id is not None
        ):
            tokenizer.pad_token_id = model.generation_config.pad_token_id
        if (
            hasattr(model.generation_config, "eos_token_id")
            and model.generation_config.eos_token_id is not None
        ):
            tokenizer.eos_token_id = model.generation_config.eos_token_id
        if (
            hasattr(model.generation_config, "bos_token_id")
            and model.generation_config.bos_token_id is not None
        ):
            tokenizer.bos_token_id = model.generation_config.bos_token_id

        if tokenizer.pad_token_id is None:
            model.generation_config.pad_token_id = (
                tokenizer.pad_token_id
            ) = tokenizer.eos_token_id

        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

        if config.device == "hpu":
            model = model.eval().to("hpu")

            if config.use_hpu_graphs and not config.use_deepspeed:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph

                model = wrap_in_hpu_graph(model)

            if config.use_deepspeed:
                model = init_deepspeed_inference(
                    model=model,
                    model_name_or_path=config.model_name_or_path,
                    use_hpu_graphs=config.use_hpu_graphs,
                )

            if config.peft_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, config.peft_path)
                model = model.to(torch.bfloat16)
        else:
            if config.peft_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, config.peft_path)
                model = model.to(torch.bfloat16)

            import intel_extension_for_pytorch as intel_ipex

            model = intel_ipex.optimize(
                model.eval(),
                dtype=torch.bfloat16,
                inplace=True,
                level="O1",
                auto_kernel_selection=True,
            )

        if not model.config.is_encoder_decoder:
            tokenizer.padding_side = "left"

        if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

        print("model loaded")
        return model, tokenizer

    def match(self, model_path: str):
        return "chatglm" in model_path.lower()

    def predict_stream(self, params):
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
            generator: A generator that yields the generated streaming text.
        """
        device = self.config.device
        prompt = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        echo = params.get("echo", True)

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        input_echo_len = len(inputs["input_ids"][0])

        gen_kwargs = {
            "max_length": max_new_tokens + input_echo_len,
            "do_sample": True if temperature > 1e-5 else False,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "logits_processor": [invalid_score_processor],
        }
        if temperature > 1e-5:
            gen_kwargs["temperature"] = temperature

        total_len = 0
        for total_ids in self.model.stream_generate(**inputs, **gen_kwargs):
            total_ids = total_ids.tolist()[0]
            total_len = len(total_ids)
            if echo:
                output_ids = total_ids
            else:
                output_ids = total_ids[input_echo_len:]
            response = self.tokenizer.decode(output_ids)
            response = self.process_response(response)

            yield response

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "chatglm2" in model_path.lower():
            return get_conv_template("chatglm2")
        return get_conv_template("chatglm")
