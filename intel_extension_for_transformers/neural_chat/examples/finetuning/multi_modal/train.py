#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import sys
import os
import ast
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import datasets

from transformers import AutoTokenizer, set_seed, BitsAndBytesConfig, AutoConfig
from transformers.integrations.deepspeed import is_deepspeed_available
from llava_utils import make_supervised_data_module, safe_save_model_for_hf_trainer
from intel_extension_for_transformers.utils.device_utils import is_hpu_available

logger = logging.getLogger(__name__)

if is_hpu_available:
    from optimum.habana import GaudiTrainingArguments as TrainingArguments
else:
    from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    pad_max: bool = False
    is_multimodal: bool = False
    mm_im_patches: int = field(default=576, metadata={"help": "for 336x336 image resolution."})
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    template: Optional[str] = field(default="v1")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    quantization_config = None
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_skip_modules=["mm_projector"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )

    low_cpu_mem_usage = True
    device_map = {"": training_args.device}
    if is_deepspeed_available():
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
        if is_deepspeed_zero3_enabled():
            low_cpu_mem_usage = False
            device_map = None

    config_kwargs = {
        "cache_dir": training_args.cache_dir,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    use_fast = True
    if config.architectures[0] == "LlamaForCausalLM":
        from intel_extension_for_transformers.transformers.modeling.llava_models.llava_llama \
                import LlavaLlamaForCausalLM
        model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
                trust_remote_code=model_args.trust_remote_code,
                use_auth_token=model_args.use_auth_token
                )
        use_fast = False
    elif config.architectures[0] == "MistralForCausalLM":
        from intel_extension_for_transformers.transformers.modeling.llava_models.llava_mistral \
                import LlavaMistralForCausalLM
        model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
                trust_remote_code=model_args.trust_remote_code,
                use_auth_token=model_args.use_auth_token,
                )
    else:
        raise ValueError("No llava implementation for the model {}".format(model_args.model_name_or_path))

    # for training
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        # Prepare the model (freeze, cast FP32, enable_require_grads, activate gradient checkpointing)
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            trust_remote_code=model_args.trust_remote_code,
            use_fast=use_fast
            )

    tokenizer.pad_token = tokenizer.unk_token

    # set vision module
    model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
            )
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    if model_args.mm_vision_select_feature == "patch":
        data_args.mm_im_patches = vision_tower.num_patches
    else:
        data_args.mm_im_patches = vision_tower.num_patches + 1
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    if data_args.image_aspect_ratio == 'anyres':
        model.config.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    if is_hpu_available:
        from optimum.habana import GaudiConfig
        gaudi_config = GaudiConfig()
        gaudi_config.use_fused_adam = True
        gaudi_config.use_fused_clip_norm = True

        from llava_trainer import GaudiLLaVATrainer
        trainer = GaudiLLaVATrainer(model=model,
                gaudi_config=gaudi_config,
                tokenizer=tokenizer,
                args=training_args,
                **data_module)
    else:
        from llava_trainer import LLaVATrainer
        trainer = LLaVATrainer(model=model,
                tokenizer=tokenizer,
                args=training_args,
                **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
