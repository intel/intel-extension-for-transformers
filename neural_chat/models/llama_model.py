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
import copy
from datetime import datetime
import torch
import logging
from threading import Thread
from typing import List
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteriaList,
)
from .utils import set_cpu_running_env, import_deepspeed, smart_context_manager
from .utils import model_is_optimized, init_deepspeed_inference, max_input_len, StopOnTokens


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class LlamaModel(BaseModel):
    def init_model(self, config):
        print("Loading model {}".format(config.model_name_or_path))
        self.config = config
        if config.device == "hpu":
            if config.use_deepspeed:
                import_deepspeed()
            # Tweak generation so that it runs faster on Gaudi
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()
        else:
            set_cpu_running_env()
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=False)

        with smart_context_manager(use_deepspeed=config.use_deepspeed):
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )

        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2

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
        start_time = datetime.now()
        device = self.config.device
        temperature = float(params["temperature"]) if "temperature" in params else 0.9
        top_p = float(params["top_p"]) if "top_p" in params else 0.75
        top_k = int(params["top_k"]) if "top_k" in params else 1
        repetition_penalty = (
            float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
        )
        max_new_tokens = (
            int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
        )
        do_sample = params["do_sample"] if "do_sample" in params else True
        num_beams = int(params["num_beams"]) if "num_beams" in params else 0
        model_name = (
            params["model_name"] if "model_name" in params else "mosaicml/mpt-7b-chat"
        )
        num_return_sequences = (
            params["num_return_sequences"] if "num_return_sequences" in params else 1
        )
        bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
        force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
        use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
        use_cache = params["use_cache"] if "use_cache" in params else True
        prompt = params["prompt"]

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        if num_beams == 0:
            num_beams = 1
            do_sample = True
        if device == "cpu":
            input_tokens = self.tokenizer.batch_encode_plus(
                [prompt], return_tensors="pt", padding=True
            )
            input_token_len = input_tokens.input_ids.shape[-1]
            if isinstance(self.model.generation_config.eos_token_id, list):
                stop_token_ids = copy.deepcopy(self.model.generation_config.eos_token_id)
            else:
                stop_token_ids = [self.model.generation_config.eos_token_id]
            end_token_id = torch.flatten(self.tokenizer("go.", return_tensors="pt").input_ids)[
                -1
            ]
            stop_token_ids.append(end_token_id)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=use_cache,
                num_return_sequences=num_return_sequences,
            )

            def generate_output():
                with torch.no_grad():
                    with torch.cpu.amp.autocast(
                        enabled=True, dtype=torch.bfloat16, cache_enabled=True
                    ):
                        generation_kwargs = dict(
                            streamer=streamer,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                        )
                        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                            [
                                StopOnTokens(
                                    min_length=max(max_new_tokens - 20, 0),
                                    start_length=input_token_len,
                                    stop_token_id=stop_token_ids,
                                )
                            ]
                        )
                        return self.model.generate(**input_tokens, **generation_kwargs)

            generation_thread = Thread(target=generate_output)
            generation_thread.start()
        elif device == "hpu":
            input_tokens = self.tokenizer.batch_encode_plus(
                [prompt],
                return_tensors="pt",
                padding="max_length",
                max_length=max_input_len(self.model, max_new_tokens),
            )
            input_token_len = input_tokens.input_ids.shape[-1]
            if isinstance(self.model.generation_config.eos_token_id, list):
                stop_token_ids = copy.deepcopy(self.model.generation_config.eos_token_id)
            else:
                stop_token_ids = [self.model.generation_config.eos_token_id]
            end_token_id = torch.flatten(self.tokenizer("go.", return_tensors="pt").input_ids)[
                -1
            ]
            stop_token_ids.append(end_token_id)
            generate_kwargs = {
                "stopping_criteria": StoppingCriteriaList(
                    [
                        StopOnTokens(
                            min_length=max(max_new_tokens - 20, 0),
                            start_length=input_token_len,
                            stop_token_id=stop_token_ids,
                        )
                    ]
                )
            }
            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(self.model.device)

            # Generation configuration
            generation_config = copy.deepcopy(self.model.generation_config)
            generation_config.max_new_tokens = max_new_tokens
            generation_config.use_cache = use_cache
            generation_config.do_sample = do_sample
            generation_config.num_beams = num_beams
            generation_config.bad_words_ids = bad_words_ids
            generation_config.force_words_ids = force_words_ids
            generation_config.num_return_sequences = num_return_sequences
            generation_config.static_shapes = model_is_optimized(self.model.config)
            generation_config.top_k = top_k
            # TODO there is an issue when top_p is used in Habana
            # generation_config.top_p = top_p
            generation_config.temperature = temperature
            generation_config.repetition_penalty = repetition_penalty

            def generate_output():
                with torch.no_grad():
                    return self.model.generate(
                        **input_tokens,
                        **generate_kwargs,
                        streamer=streamer,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                        lazy_mode=True,
                        hpu_graphs=use_hpu_graphs,
                        ignore_eos=False,
                    )

            generation_thread = Thread(target=generate_output)
            generation_thread.start()
        else:
            raise ValueError(
                f"Unsupported device type {device}, only supports cpu and hpu now."
            )
        output_word_len = 0

        for new_text in streamer:
            if len(new_text) == 0:
                continue
            if output_word_len == 0:
                first_token_output_time = datetime.now()
            output_word_len += 1
            yield new_text

        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)
        first_word_latency = int(
            (first_token_output_time - start_time).total_seconds() * 1000
        )
        msecond_per_word = (
            (duration - first_word_latency) / (output_word_len - 1)
            if output_word_len != 1
            else 0
        )
        stats = {
            "input_token_len": input_token_len,
            "output_word_len": output_word_len,
            "duration": duration,
            "first_word_latency": first_word_latency,
            "msecond_per_word": msecond_per_word,
        }
        yield "END_OF_STREAM_STATS={}".format(stats)

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
        device = self.config.device
        temperature = float(params["temperature"]) if "temperature" in params else 0.9
        top_p = float(params["top_p"]) if "top_p" in params else 0.75
        top_k = int(params["top_k"]) if "top_k" in params else 1
        repetition_penalty = (
            float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
        )
        max_new_tokens = (
            int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
        )
        do_sample = params["do_sample"] if "do_sample" in params else True
        num_beams = int(params["num_beams"]) if "num_beams" in params else 0
        model_name = (
            params["model_name"] if "model_name" in params else "mosaicml/mpt-7b-chat"
        )
        num_return_sequences = (
            params["num_return_sequences"] if "num_return_sequences" in params else 1
        )
        bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
        force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
        use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
        use_cache = params["use_cache"] if "use_cache" in params else False
        prompt = params["prompt"]
        if num_beams == 0:
            num_beams = 1
            do_sample = True
        if device == "cpu":
            input_tokens = self.tokenizer.batch_encode_plus(
                [prompt], return_tensors="pt", padding=True
            )
            input_token_len = input_tokens.input_ids.shape[-1]
            if isinstance(self.model.generation_config.eos_token_id, list):
                stop_token_ids = copy.deepcopy(self.model.generation_config.eos_token_id)
            else:
                stop_token_ids = [self.model.generation_config.eos_token_id]
            end_token_id = torch.flatten(self.tokenizer("go.", return_tensors="pt").input_ids)[
                -1
            ]
            stop_token_ids.append(end_token_id)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=use_cache,
                num_return_sequences=num_return_sequences,
            )

            with torch.no_grad():
                with torch.cpu.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
                ):
                    generation_kwargs = dict(
                        generation_config=generation_config, return_dict_in_generate=True
                    )
                    generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [
                            StopOnTokens(
                                min_length=max(max_new_tokens - 20, 0),
                                start_length=input_token_len,
                                stop_token_id=stop_token_ids,
                            )
                        ]
                    )
                    generation_output = self.model.generate(**input_tokens, **generation_kwargs)
        elif device == "hpu":
            input_tokens = self.tokenizer.batch_encode_plus(
                [prompt],
                return_tensors="pt",
                padding="max_length",
                max_length=max_input_len(self.model, max_new_tokens),
            )
            input_token_len = input_tokens.input_ids.shape[-1]
            if isinstance(self.model.generation_config.eos_token_id, list):
                stop_token_ids = copy.deepcopy(self.model.generation_config.eos_token_id)
            else:
                stop_token_ids = [self.model.generation_config.eos_token_id]
            end_token_id = torch.flatten(self.tokenizer("go.", return_tensors="pt").input_ids)[
                -1
            ]
            stop_token_ids.append(end_token_id)
            generate_kwargs = {
                "stopping_criteria": StoppingCriteriaList(
                    [
                        StopOnTokens(
                            min_length=max(max_new_tokens - 20, 0),
                            start_length=input_token_len,
                            stop_token_id=stop_token_ids,
                        )
                    ]
                )
            }
            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(self.model.device)

            # Generation configuration
            generation_config = copy.deepcopy(self.model.generation_config)
            generation_config.max_new_tokens = max_new_tokens
            generation_config.use_cache = use_cache
            generation_config.do_sample = do_sample
            generation_config.num_beams = num_beams
            generation_config.bad_words_ids = bad_words_ids
            generation_config.force_words_ids = force_words_ids
            generation_config.num_return_sequences = num_return_sequences
            generation_config.static_shapes = model_is_optimized(self.model.config)
            generation_config.top_k = top_k
            # TODO there is an issue when top_p is used in Habana
            # generation_config.top_p = top_p
            generation_config.temperature = temperature
            generation_config.repetition_penalty = repetition_penalty

            with torch.no_grad():
                generation_output = self.model.generate(
                    **input_tokens,
                    **generate_kwargs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    lazy_mode=True,
                    hpu_graphs=use_hpu_graphs,
                    ignore_eos=False,
                )
        output = self.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        if "### Response:" in output:
            return output.split("### Response:")[1].strip()
        return output
