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
from typing import Optional, List, Dict
from transformers import TrainingArguments
from transformers.utils.versions import require_version
from dataclasses import dataclass

from neural_chat.pipeline.plugins.audio.asr import AudioSpeechRecognition
from neural_chat.pipeline.plugins.audio.asr_chinese import ChineseAudioSpeechRecognition
from neural_chat.pipeline.plugins.audio.tts import TextToSpeech
from neural_chat.pipeline.plugins.audio.tts_chinese import ChineseTextToSpeech
from .plugins import plugins

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

class AudioLanguageOptions(Enum):
    ENGLISH = auto()
    CHINESE = auto()

class RetrievalTypeOptions(Enum):
    SPARSE = auto()
    DENSE = auto()

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

@dataclass
class FinetuningConfig:
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    finetune_args: FinetuningArguments

@dataclass
class GenerationConfig:
    device: str = "cpu"
    temperature: float = 0.9
    top_k: int = 1
    top_p: float = 0.75
    repetition_penalty: float = 1.1
    num_beams: int = 0
    max_new_tokens: int = 256
    do_sample: bool = True
    num_return_sequences: int = 1
    bad_words_ids: List[int] = None
    force_words_ids: List[int] = None
    use_hpu_graphs: bool = False
    use_cache: bool = False
    audio_output_path: str = None
    cpu_jit: bool = False
    num_gpus: int = 0
    max_gpu_memory: int = None
    use_fp16: bool = False
    ipex_int8: bool = False

@dataclass
class LoadingModelConfig:
    cpu_jit: bool = None
    peft_path: str = None
    use_hpu_graphs: bool = False
    use_cache: bool = False
    use_deepspeed: bool = False

@dataclass
class WeightOnlyQuantizationConfig:
    algorithm: str = 'RTN'
    bits: int = 8
    group_size: int = -1
    scheme: str = 'sym'
    sym_full_range: bool = True

@dataclass
class AMPConfig:
    dtype: str = 'bfloat16'

@dataclass
class RetrieverConfig:
    search_type: str = "mmr"
    search_kwargs: Dict[str, int] = field(default=lambda: {"k": 1, "fetch_k": 5})
    retrieval_topk: int = 1
    
@dataclass
class SafetyConfig:
    dict_path: str = None
    matchType: int = 2

@dataclass
class OptimizationConfig:
    amp_config: AMPConfig = AMPConfig()
    weight_only_quant_config: WeightOnlyQuantizationConfig = None
    
@dataclass
class IntentConfig:
    max_new_tokens: int = 5
    temperature: float = 0.6
    do_sample: bool = True
    top_k: int = 1
    repetition_penalty:float = 1.0
    num_return_sequences: int = 1
    bad_words_ids: List[int] = None
    force_words_ids: List[int] = None
    use_hpu_graphs: bool = False
    use_cache: bool = False
    audio_output_path: str = None
    cpu_jit: bool = False
    num_gpus: int = 0
    max_gpu_memory: int = None
    use_fp16: bool = False
    ipex_int8: bool = False


class PipelineConfig:
    def __init__(self,
                 model_name_or_path="meta-llama/Llama-2-7b-hf",
                 tokenizer_name_or_path=None,
                 device="auto",
                 plugin=plugins,
                 loading_config=None,
                 optimization_config=None):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.device = device
        self.plugin = plugin
        self.loading_config = loading_config if loading_config is not None else LoadingModelConfig()
        self.optimization_config = optimization_config if optimization_config is not None else OptimizationConfig()
        for plugin_name, plugin_value in self.plugin.items():
            if plugin_value['enable']:
                print(f"create {plugin_name} plugin instance...")
                print(f"plugin parameters: ", plugin_value['args'])
                plugins[plugin_name]["instance"] = plugin_value['class'](**plugin_value['args'])



