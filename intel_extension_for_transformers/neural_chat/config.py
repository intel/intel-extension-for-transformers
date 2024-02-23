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
from transformers import TrainingArguments
from transformers.utils.versions import require_version
from dataclasses import dataclass
from .utils.common import get_device_type

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
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34."
            "Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "should enable when using custom model architecture that is not yet part of "
                    "the Hugging Face transformers package like MPT)."
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
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. "
                    "This seed and the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, "
                    "try to load processed dataset from this path."
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
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
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
            "help": "The length of the virtual tokens to insert in P-tuning/Prompt-tuning/Prefix-tuning"
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
        default="auto",
        metadata={
            "help": "What device to use for finetuning.",
            "choices": ["cpu", "cuda", "hpu", "auto"],
        },
    )
    lora_all_linear: bool = field(
        default=False,
        metadata={"help": "if True, will add adaptor for all linear for lora finetuning"},
    )
    task: Optional[str] = field(
        default="completion",
        metadata={"help": "task name, different task means different data format.",
            "choices": ["completion", "chat", "summarization", "code-generation"]
            },
    )
    do_lm_eval: bool = field(
        default=False,
        metadata={"help": "whether to run the LM evaluation with EleutherAI/lm-evaluation-harness"},
    )
    lm_eval_tasks: Optional[List[str]] = field(
        default_factory=lambda: ["truthfulqa_mc"],
        metadata={"help": "tasks list for accuracy validation with EleutherAI/lm-evaluation-harness."},
    )
    qlora: bool = field(
        default=False,
        metadata={"help": "whether use qlora for finetuning"},
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
        default=4,
        metadata={"help": "How many bits to use."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )

@dataclass
class TTSDatasetArguments:
    audio_folder_path: Optional[str] = field(default=None, metadata={"help": "The path to the directory of audios."})
    text_folder_path: Optional[str] = field(default=None, metadata={"help": "The path to the directory of texts."})

@dataclass
class TTSModelArguments:
    step: int = field(default=0, metadata={"help": "TTS model step."})
    warmup_step: int = field(default=0, metadata={"help": "TTS model warmup step."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate."})

@dataclass
class BaseFinetuningConfig:
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    finetune_args: FinetuningArguments

TextGenerationFinetuningConfig = BaseFinetuningConfig

SummarizationFinetuningConfig = BaseFinetuningConfig

CodeGenerationFinetuningConfig = BaseFinetuningConfig

@dataclass
class TTSFinetuningConfig:
    dataset_args: TTSDatasetArguments
    model_args: TTSModelArguments

@dataclass
class GenerationConfig:
    device: str = "cpu"
    temperature: float = 0.1
    top_k: int = 40
    top_p: float = 0.75
    repetition_penalty: float = 1.1
    num_beams: int = 1
    max_new_tokens: int = 256
    do_sample: bool = True
    num_return_sequences: int = 1
    bad_words_ids: List[int] = None
    force_words_ids: List[int] = None
    use_hpu_graphs: bool = False
    use_cache: bool = True
    audio_output_path: str = None
    cpu_jit: bool = False
    num_gpus: int = 0
    max_gpu_memory: int = None
    use_fp16: bool = False
    ipex_int8: bool = False
    return_stats: bool = False
    format_version: str = "v2"
    task: str = ""
    sql_metadata: str = ""

@dataclass
class LoadingModelConfig:
    cpu_jit: bool = None
    peft_path: str = None
    use_hpu_graphs: bool = False
    use_cache: bool = True
    use_deepspeed: bool = False
    world_size: int = 1
    ipex_int8: bool = False
    use_neural_speed: bool = False
    gguf_model_path: str = None

@dataclass
class FrameworkConfig:
    pass

@dataclass
class VllmEngineParams(FrameworkConfig):
    # to use continuous batching during serving, use_async_engine should be set true,
    # otherwise, serving is offline and synchronous, which means the next batch will only
    # be queued and processed after the processing of the last batch is finished
    use_async_engine: bool = True
    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py
    tensor_parallel_size: int = 1
    quantization: str = None
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192

@dataclass
class ServingConfig:
    framework: str = "vllm" # vllm/TGI
    framework_config: FrameworkConfig = None

class PipelineConfig:
    def __init__(self,
                 model_name_or_path="Intel/neural-chat-7b-v3-1",
                 tokenizer_name_or_path=None,
                 hf_access_token=None,
                 device="auto",
                 task="",
                 plugins=plugins,
                 loading_config=None,
                 optimization_config=None,
                 assistant_model=None,
                 serving_config=None):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.hf_access_token = hf_access_token
        if device == "auto":
            self.device = get_device_type()
        else:
            self.device = device
        self.task = task
        self.plugins = plugins

        self.loading_config = loading_config if loading_config is not None else \
            LoadingModelConfig(cpu_jit=True if self.device == "cpu" else False, \
                use_hpu_graphs = True if self.device == "hpu" else False)
        from intel_extension_for_transformers.transformers import (
            MixedPrecisionConfig,
            WeightOnlyQuantConfig,
            BitsAndBytesConfig
        )
        self.optimization_config = optimization_config if optimization_config is not None else \
            MixedPrecisionConfig(dtype="float16" if self.device == "cuda" else "bfloat16")
        assert type(self.optimization_config) in [MixedPrecisionConfig, WeightOnlyQuantConfig, BitsAndBytesConfig], \
            f"Expect optimization_config be an object of MixedPrecisionConfig, WeightOnlyQuantConfig" + \
            " or BitsAndBytesConfig,got {type(self.optimization_config)}."
        self.assistant_model = assistant_model
        self.serving_config = serving_config
