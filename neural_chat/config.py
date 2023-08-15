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
"""Configs for Neural Chat."""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from transformers import TrainingArguments
from transformers.utils.versions import require_version

from enum import Enum, auto

class DeviceOptions(Enum):
    AUTO = auto()
    CPU = auto()
    GPU = auto()
    XPU = auto()
    HPU = auto()
    CUDA = auto()

class BackendOptions(Enum):
    AUTO = auto()
    TORCH = auto()
    IPEX = auto()
    ITREX = auto()

class AudioOptions(Enum):
    ENGLISH = auto()
    CHINESE = auto()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
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
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "should enable when using custom model architecture that is not yet part of the Hugging Face transformers package like MPT)."
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=0,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep in memory the loaded dataset. Defaults to False."
        },
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, try to load processed dataset from this path."
        },
    )
    dataset_concatenation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to concatenate the sentence for more efficient training."
        },
    )
    special_tokens: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The list of special tokens to add in tokenizer."}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


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
    adapter_layers: int = field(
        default=30,
        metadata={"help": "adapter layer number in the LLaMA-adapter."},
    )
    adapter_len: int = field(
        default=10,
        metadata={
            "help": "The length of the adaption prompt to insert in the LLaMA-adapter."
        },
    )
    num_virtual_tokens: int = field(
        default=10,
        metadata={
            "help": "The length of the vitrual tokens to insert in P-tuning/Prompt-tuning/Prefix-tuning"
        },
    )
    ptun_hidden_size: int = field(
        default=1024,
        metadata={"help": "The encoder hidden size in P-tuning"},
    )
    peft: Optional[str] = field(
        default="lora",
        metadata={
            "help": ("apply peft. default set to lora"),
            "choices": ["llama_adapter", "lora", "ptun", "prefix", "prompt"],
        },
    )
    train_on_inputs: bool = field(
        default=False,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    device: str = field(
        default="cpu",
        metadata={
            "help": "What device to use for finetuning.",
            "choices": ["cpu", "cuda", "habana", "auto"],
        },
    )
    lora_all_linear: bool = field(
        default=False,
        metadata={"help": "if True, will add adaptor for all linear for lora finetuning"},
    )


class FinetuningConfig:
    def __init__(self,
                 model_args: ModelArguments,
                 data_args: DataArguments,
                 training_args: TrainingArguments,
                 finetune_args: FinetuningArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.finetune_args = finetune_args


class OptimizationConfig:
    def __init__(self,
                 mode='latency',
                 device='cpu',
                 backend='ipex',
                 approach="static",
                 precision='bf16',
                 excluded_precisions=[],
                 op_type_dict=None,
                 op_name_dict=None,
                 recipes={}):
        self.mode = mode
        self.device = device
        self.backend = backend
        self.approach = approach
        self.precision = precision
        self.excluded_precisions = excluded_precisions
        self.op_type_dict = op_type_dict
        self.op_name_dict = op_name_dict
        self.recipes = recipes


class NeuralChatConfig:
    def __init__(self,
                 model_name_or_path="meta-llama/Llama-2-70b-hf",
                 tokenizer_name_or_path=None,
                 device="auto",
                 backend="auto",
                 retrieval=False,
                 retrieval_type=None,
                 document_path=None,
                 audio_input=False,
                 audio_input_path=None,
                 audio_output=False,
                 audio_output_path=False,
                 audio_lang=None,
                 txt2Image=False,
                 server_mode=True,
                 use_hpu_graphs=False,
                 use_deepspeed=False,
                 peft_path=None,
                 cpu_jit=False,
                 use_cache=False,
                 num_gpus=0,
                 max_gpu_memory=None,
                 cache_chat=False,
                 cache_chat_config_file=None,
                 cache_embedding_model_dir=None,
                 intent_detection=False,
                 memory_controller=False,
                 safety_checker=False):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.device = device
        self.backend = backend
        self.retrieval = retrieval
        self.retrieval_type = retrieval_type
        self.document_path = document_path
        self.audio_input = audio_input
        self.audio_input_path = audio_input_path
        self.audio_output = audio_output
        self.audio_output_path = audio_output_path
        self.audio_lang = audio_lang
        self.txt2Image = txt2Image
        self.server_mode = server_mode
        self.use_hpu_graphs = use_hpu_graphs
        self.use_deepspeed = use_deepspeed
        self.peft_path = peft_path
        self.cpu_jit = cpu_jit
        self.use_cache = use_cache
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.cache_chat = cache_chat
        self.cache_chat_config_file = cache_chat_config_file
        self.cache_embedding_model_dir = cache_embedding_model_dir
        self.intent_detection = intent_detection
        self.memory_controller = memory_controller
        self.safety_checker = safety_checker
