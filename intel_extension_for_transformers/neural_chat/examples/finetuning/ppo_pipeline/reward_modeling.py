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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
import logging
import sys
from intel_extension_for_transformers.neural_chat.utils.common import is_hpu_available

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


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
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    hf_access_token: str = field(
        default=None,
        metadata={"help": "Huggingface token to access model."},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "whether to use load_in_4bit"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="Intel/orca_dpo_pairs",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
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
    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
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
        default=32,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    lora_all_linear: bool = field(
        default=True,
        metadata={
            "help": "if True, will add adaptor for all linear for lora finetuning"
        },
    )


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for system, question, response_j, response_k in zip(
        examples["system"], examples["question"], examples["chosen"], examples["rejected"]
    ):
        tokenized_j = tokenizer(
            system + question + response_j, truncation=True
        )
        tokenized_k = tokenizer(
            system + question + response_k, truncation=True
        )

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
def compute_loss(self, model, inputs, return_outputs=False):
    rewards_j = model(
        input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"]
    )[0]
    rewards_k = model(
        input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"]
    )[0]
    loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
    if return_outputs:
        return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
    return loss


if __name__ == "__main__":
    if not is_hpu_available:
        from transformers import set_seed

        parser = HfArgumentParser(
            (
                ModelArguments,
                DataTrainingArguments,
                TrainingArguments,
                FinetuningArguments,
            )
        )
    else:
        from optimum.habana import GaudiTrainingArguments, GaudiTrainer
        from optimum.habana.utils import set_seed

        parser = HfArgumentParser(
            (
                ModelArguments,
                DataTrainingArguments,
                GaudiTrainingArguments,
                FinetuningArguments,
            )
        )
    (
        model_args,
        data_args,
        training_args,
        finetune_args,
    ) = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    training_args.label_names = []
    set_seed(training_args.seed)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load the human stack-exchange-paired dataset for tuning the reward model.
    train_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_access_token,
        split=f"train[{data_args.validation_split_percentage}%:]",
    )
    eval_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_access_token,
        split=f"train[:{data_args.validation_split_percentage}%]",
    )

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path
    )
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.hf_access_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=finetune_args.lora_alpha,
        lora_dropout=finetune_args.lora_dropout,
        modules_to_save=["score"],
    )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    # load policy model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=model_args.load_in_4bit,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.hf_access_token,
        trust_remote_code=True,
    )
    if model.score.weight.is_meta:
        model.score.weight = torch.nn.parameter.Parameter(
            data=torch.randn(
                model.score.weight.shape,
                device=model.device,
                dtype=model.score.weight.dtype,
            ),
            requires_grad=model.score.weight.data.requires_grad,
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model = get_peft_model(model, peft_config)
    model = model.to(torch_dtype)

    model.print_trainable_parameters()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not training_args.gradient_checkpointing
    num_proc = (
        data_args.preprocessing_num_workers
    )  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names

    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= data_args.max_length
        and len(x["input_ids_k"]) <= data_args.max_length
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= data_args.max_length
        and len(x["input_ids_k"]) <= data_args.max_length
    )

    # Define the metric that we'll use for validation.
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # Here, predictions is rewards_j and rewards_k.
        # We want to see how much of the time rewards_j > rewards_k.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)

    # Train the model, woohoo.
    if hasattr(training_args, "use_habana"):
        from optimum.habana import GaudiTrainer, GaudiConfig

        gaudi_config = GaudiConfig()
        gaudi_config.use_fused_adam = True
        gaudi_config.use_fused_clip_norm = True
        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=RewardDataCollatorWithPadding(
                tokenizer=tokenizer, max_length=data_args.max_length
            ),
        )
    else:
        from transformers import Trainer

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=RewardDataCollatorWithPadding(
                tokenizer=tokenizer, max_length=data_args.max_length
            ),
        )
    trainer.__class__.compute_loss = compute_loss

    trainer.train()

    trainer.model = trainer.model.merge_and_unload()
    trainer.save_model()
