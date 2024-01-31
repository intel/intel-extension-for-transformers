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

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
)

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from intel_extension_for_transformers.utils.device_utils import is_hpu_available

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=2,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    pad_max: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class FinetuningArguments:
    """
    Arguments of finetune we are going to apply on the model.
    """

    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank parameter in the LoRA method."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    lora_all_linear: bool = field(
        default=True,
        metadata={"help": "if True, will add adaptor for all linear for lora finetuning"},
    )
    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


if __name__ == "__main__":

    if not is_hpu_available:
        from transformers import set_seed
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, FinetuningArguments))
        load_in_4bit = True
    else:
        from optimum.habana import GaudiTrainingArguments
        from optimum.habana.utils import set_seed
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, GaudiTrainingArguments, FinetuningArguments)
        )

        # not support bisandbytes currently
        load_in_4bit = False

    model_args, data_args, training_args, finetune_args = parser.parse_args_into_dataclasses()

    if training_args.use_cpu:
        load_in_4bit = False

    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        streaming=data_args.streaming,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )


    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [system + question for system,question in zip(samples["system"], samples["question"])],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    column_names = raw_datasets["train"].column_names

    raw_datasets = raw_datasets.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    # model config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
            )

    # load policy model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )
    model.config.use_cache = False

    # load reference model
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )
    model_ref.config.use_cache = False

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token


    # Training preprocessing
    def prepare_features(examples):

        prompts = {p.strip() for p in examples["prompt"]}
        chosens = {c.strip() for c in examples["chosen"]}
        rejects = {r.strip() for r in examples["rejected"]}

        examples = {
                "prompt": [],
                "chosen": [],
                "rejected": [],
                "chosen_response_only": [],
                "rejected_response_only": [],
                "chosen_input_ids": [],
                "chosen_attention_mask": [],
                "chosen_labels": [],
                "rejected_input_ids": [],
                "rejected_attention_mask": [],
                "rejected_labels": [],
                "prompt_input_ids": [],
                "prompt_attention_mask": []}

        for prompt, chosen, reject in zip(prompts, chosens, rejects):

            prompt_tokens = tokenizer.tokenize(prompt)

            if len(prompt_tokens) > data_args.max_prompt_length:
                prompt_tokens = prompt_tokens[:data_args.max_prompt_length]

            prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
            prompt_mask = [1] * len(prompt_ids)

            max_resp = data_args.max_length - len(prompt_ids)
            chosen_tokens = tokenizer.tokenize(chosen)
            chosen_tokens = chosen_tokens[:max_resp - 1]
            chosen_tokens.append(tokenizer.eos_token)
            chosen_ids = tokenizer.convert_tokens_to_ids(chosen_tokens)
            chosen_mask = [1] * len(chosen_ids)

            reject_tokens = tokenizer.tokenize(reject)
            reject_tokens = reject_tokens[:max_resp - 1]
            reject_tokens.append(tokenizer.eos_token)
            reject_ids = tokenizer.convert_tokens_to_ids(reject_tokens)
            reject_mask = [1] * len(reject_ids)

            chosen_input_ids = prompt_ids + chosen_ids
            chosen_attention_mask = prompt_mask + chosen_mask
            chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids

            reject_input_ids = prompt_ids + reject_ids
            reject_attention_mask = prompt_mask + reject_mask
            reject_labels = [IGNORE_INDEX] * len(prompt_ids) + reject_ids

            # padding
            input_len = len(chosen_input_ids)
            if data_args.pad_max:
                pad_len = data_args.max_length - input_len
                chosen_input_ids = chosen_input_ids + [0] * pad_len
                chosen_labels = chosen_labels + [-100] * pad_len
                chosen_attention_mask = chosen_attention_mask + [0] * pad_len
                assert len(chosen_input_ids) == data_args.max_length

            input_len = len(reject_input_ids)
            if data_args.pad_max:
                pad_len = data_args.max_length - input_len
                reject_input_ids = reject_input_ids + [0] * pad_len
                reject_labels = reject_labels + [-100] * pad_len
                reject_attention_mask = reject_attention_mask + [0] * pad_len
                assert len(reject_input_ids) == data_args.max_length

            examples["prompt"].append(prompt)
            examples["chosen"].append(prompt + chosen)
            examples["rejected"].append(prompt + reject)
            examples["chosen_response_only"].append(chosen)
            examples["rejected_response_only"].append(reject)

            examples["chosen_input_ids"].append(chosen_input_ids)
            examples["chosen_attention_mask"].append(chosen_attention_mask)
            examples["chosen_labels"].append(chosen_labels)

            examples["rejected_input_ids"].append(reject_input_ids)
            examples["rejected_attention_mask"].append(reject_attention_mask)
            examples["rejected_labels"].append(reject_labels)

            examples["prompt_input_ids"].append(prompt_ids)
            examples["prompt_attention_mask"].append(prompt_mask)

        return examples

    train_dataset = raw_datasets["train"]
    column_names = train_dataset.column_names
    # Create train feature from dataset
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    eval_examples = raw_datasets["validation"]
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_examples.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    def collate_fn(batch):
        input_ids = [torch.tensor(ins["chosen_input_ids"]) for ins in batch] +\
                [torch.tensor(ins["rejected_input_ids"]) for ins in batch]
        labels = [torch.tensor(ins["chosen_labels"]) for ins in batch] +\
                [torch.tensor(ins["rejected_labels"]) for ins in batch]
        attention_mask = [torch.tensor(ins["chosen_attention_mask"]) for ins in batch] +\
                [torch.tensor(ins["rejected_attention_mask"]) for ins in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    if finetune_args.lora_all_linear:
        target_modules = find_all_linear_names(model)
    elif finetune_args.lora_target_modules is not None:
        target_modules = finetune_args.lora_target_modules
    else:
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
            "Wqkv"
        ]

    peft_config = LoraConfig(
        r=finetune_args.lora_rank,
        lora_alpha=finetune_args.lora_alpha,
        lora_dropout=finetune_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    if not hasattr(training_args, "use_habana"):
        from intel_extension_for_transformers.transformers.dpo_trainer import DPOTrainer
    else:
        from intel_extension_for_transformers.transformers.dpo_trainer import GaudiDPOTrainer as DPOTrainer

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        data_collator=collate_fn,
        beta=finetune_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=data_args.max_length,
    )

    # 6. train
    dpo_trainer.train()

    # 7. save the model
    dpo_trainer.save_model(training_args.output_dir)
