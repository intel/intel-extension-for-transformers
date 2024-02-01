# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import logging
from typing import Dict, List, Optional, Type, Union
from utils import detect_language

import torch
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.prompt.prompt_node import PromptModel, PromptModelInvocationLayer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import intel_extension_for_pytorch as ipex
logger = logging.getLogger(__name__)

class StopOnTokensWithPeriod(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: list[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if scores is not None:
            if len(scores) > self.min_length:
                for stop_id in self.stop_token_id:
                    if input_ids[0][self.start_length - 1 + len(scores)] == stop_id:
                        return True
        elif input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False

class TransformersDecoderInvocationLayer(PromptModelInvocationLayer):
    def __init__(
        self,
        model_name_or_path: str = "mpt-7b-chat",
        max_length: Optional[int] = 256,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        **kwargs,
    ):
        super().__init__(model_name_or_path, max_length)
        self.use_auth_token = use_auth_token

        self.devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        self.use_gpu = use_gpu
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "model_kwargs",
                "trust_remote_code",
                "revision",
                "feature_extractor",
                "tokenizer",
                "config",
                "use_fast",
                "torch_dtype",
                "device_map",
            ]
            if key in kwargs
        }
        # flatten model_kwargs one level
        if "model_kwargs" in model_input_kwargs:
            mkwargs = model_input_kwargs.pop("model_kwargs")
            model_input_kwargs.update(mkwargs)

        torch_dtype = model_input_kwargs.get("torch_dtype")
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if "torch." not in torch_dtype:
                    raise ValueError(
                        f"torch_dtype should be a torch.dtype or a string with 'torch.' prefix, got {torch_dtype}"
                    )
                torch_dtype_resolved = getattr(torch, torch_dtype.strip("torch."))
            elif isinstance(torch_dtype, torch.dtype):
                torch_dtype_resolved = torch_dtype
            else:
                raise ValueError(f"Invalid torch_dtype value {torch_dtype}")
            model_input_kwargs["torch_dtype"] = torch_dtype_resolved

        if len(model_input_kwargs) > 0:
            logger.info(
                "Using model input kwargs %s in %s", model_input_kwargs, self.__class__.__name__
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    low_cpu_mem_usage=True,
                                                    return_dict=True,
                                                    torch_dtype=torch.bfloat16,
                                                    max_seq_len=8192,
                                                    trust_remote_code=True)
        self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
        if self.use_gpu:
            self.model = self.model.to("cuda")

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model = self.model.eval()

    def invoke(self, *args, **kwargs):
        # generate config default values
        generate_kwargs = dict(
            use_cache=True,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )

        generate_kwargs.update(
            dict(
                min_new_tokens=kwargs.pop("min_new_tokens", 1),
                max_new_tokens=kwargs.pop("max_new_tokens", self.max_length),
                temperature=kwargs.pop("temperature", 0.3),
                top_p=kwargs.pop("top_p", 0.9),
                top_k=kwargs.pop("top_k", 1),
                num_beams=kwargs.pop("beams", 1),
                return_dict_in_generate=True,
                early_stopping=kwargs.pop("early_stopping", True),
            )
        )
        decode_mode = (kwargs.pop("decode_mode", "Greedy"),)
        if "Greedy" in decode_mode:
            generate_kwargs.update(dict(num_beams=1, do_sample=False))
        elif "Beam" in decode_mode:
            generate_kwargs.update(dict(do_sample=True))
        else:
            pass

        output: List[Dict[str, str]] = []
        start_time = time.time()
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")
            if detect_language(prompt) == 'Chinese':
                generate_kwargs["max_new_tokens"] = 512
            input_tokens = self.tokenizer.batch_encode_plus(
                [prompt], return_tensors="pt", padding=True
            )
            input_token_len = input_tokens.input_ids.shape[-1]
            stop_token_ids = self.tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
            stop_token_ids.append(self.model.generation_config.eos_token_id)
            stop_token_ids.append(self.tokenizer(".", return_tensors="pt").input_ids)
            stop_token_ids.append(self.tokenizer("!", return_tensors="pt").input_ids)
            stop_token_ids.append(self.tokenizer("。", return_tensors="pt").input_ids)
            stop_token_ids.append(self.tokenizer("！", return_tensors="pt").input_ids)
            stop = StopOnTokensWithPeriod(min_length=108, start_length=input_token_len, stop_token_id=stop_token_ids)
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
            with torch.no_grad():
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                    output = self.model.generate(**input_tokens, **generate_kwargs)
        generated_texts = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        print("The inference time======", time.time() - start_time)
        if "Response:" in generated_texts:
            result =  generated_texts.split("Response:")[1].strip()
        return [result]

    @classmethod
    def supports(cls, model_name_or_path: str) -> bool:
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        except OSError:
            # This is needed so OpenAI models are skipped over
            return False
        supported_models = list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        supported_models.append("MPTForCausalLM")
        return config.architectures[0] in supported_models


class MptPromptModel(PromptModel):
    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "mpt-7b-chat",
        max_length: Optional[int] = 256,
        api_key: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.api_key = api_key
        self.use_auth_token = use_auth_token
        self.use_gpu = use_gpu
        self.devices = devices

        self.model_kwargs = model_kwargs if model_kwargs else {}

        self.invocation_layers: List[Type[PromptModelInvocationLayer]] = []

        self.register(TransformersDecoderInvocationLayer)

        self.model_invocation_layer = self.create_invocation_layer()

    def invoke(self, prompt: Union[str, List[str]], **kwargs) -> List[str]:
        output = self.model_invocation_layer.invoke(prompt=prompt, **kwargs, **self.model_kwargs)
        return output
