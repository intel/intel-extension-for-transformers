#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
    AutoModelForSequenceClassification,
)

from intel_extension_for_transformers.transformers.ppo_core import (
    LengthSampler,
    set_seed,
)
from intel_extension_for_transformers.transformers.ppo_config import PPOConfig
from intel_extension_for_transformers.transformers.ppo_trainer import PPOTrainer
from intel_extension_for_transformers.transformers.modeling.trl_models import (
    AutoModelForCausalLMWithValueHead,
)

import sys
import logging
import time

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def build_dataset(
    tokenizer,
    dataset_name,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for system, question in zip(examples["system"], examples["question"]):
            query = system + question
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(
        default="", metadata={"help": "the tokenizer name"}
    )
    reward_model_name: Optional[str] = field(
        default="", metadata={"help": "the reward model name"}
    )
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    output_max_length: Optional[int] = field(
        default=128, metadata={"help": "maximum length for generation"}
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"}
    )
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "the number of ppo epochs"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the adafactor optimizer"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the batched text gen"}
    )
    save_freq: Optional[int] = field(
        default=None, metadata={"help": "n steps to save the model"}
    )
    output_dir: Optional[str] = field(
        default="runs/", metadata={"help": "n steps to save the model"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Initial KL penalty coefficient (used for adaptive and linear control)"
        },
    )

    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"}
    )

    hf_access_token: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface token to access model."},
    )

    dataset_name: Optional[str] = field(
        default="Intel/orca_dpo_pairs",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    lora_rank: Optional[int] = field(
        default=16,
        metadata={"help": "Rank parameter in the LoRA method."},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    use_habana: Optional[bool] = field(
        default=False, metadata={"help": "use habana for RL training"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    reward_model_name = script_args.reward_model_name
    config = PPOConfig(
        steps=script_args.steps,
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_device_cache=False,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        use_habana=script_args.use_habana,
        pad_for_acceleration=script_args.use_habana,
        pad_max_len=512 + script_args.output_max_length,
        pad_max_input_len=512,
    )

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
        "truncation": True,
    }

    if config.pad_for_acceleration:
        sent_kwargs["padding"] = "max_length"
        # is 1024 enough?
        sent_kwargs["max_length"] = 1024

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name, token=script_args.hf_access_token
    )

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer, dataset_name=script_args.dataset_name)

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    lora_config = LoraConfig(
        r=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        peft_config=lora_config,
        token=script_args.hf_access_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(torch.bfloat16)

    if script_args.use_habana:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            token=script_args.hf_access_token,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        ref_model = None

    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the sentiment analysis pipeline using our reward model, passing the
    # model name and the sentiment analysis pipeline arguments. Let's also make sure to
    # set the device to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        if torch.cuda.is_available():
            device = 0

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=script_args.hf_access_token,
    )

    if config.use_habana:
        from habana_frameworks.torch.hpu import (
            wrap_in_hpu_graph,
        )  # pylint: disable=E0611, E0401

        reward_model = wrap_in_hpu_graph(reward_model)

    if device.type == "hpu":
        device = "hpu"

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model,
        tokenizer=tokenizer,
        return_token_type_ids=False,
        device=device,
        model_kwargs={
            "use_auth_token": script_args.hf_access_token,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
        },
    )

    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if sentiment_pipe.tokenizer.pad_token_id is None:
        sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 32
    output_max_length = script_args.output_max_length
    if not config.pad_for_acceleration:
        output_length_sampler = LengthSampler(output_min_length, output_max_length)
    else:
        output_length_sampler = LengthSampler(output_max_length, output_max_length + 1)

    epochs = tqdm(
        enumerate(ppo_trainer.dataloader),
        total=len(ppo_trainer.dataloader),
        desc="rl progress",
    )
    for epoch, batch in epochs:
        if epoch >= config.total_ppo_epochs:
            break

        question_tensors = batch["input_ids"]
        t0 = time.time()
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )
        t1 = time.time()
        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [
            torch.tensor(output[0]["score"] - script_args.reward_baseline)
            for output in pipe_outputs
        ]
        t2 = time.time()
        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        stats["time/ppo/rollout"] = t1-t0
        stats["time/ppo/evaluate"] = t2-t1
        ppo_trainer.log_stats(stats, batch, rewards)
        epochs.update(1)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
