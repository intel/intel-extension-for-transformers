#!/usr/bin/env python
# coding=utf-8

# Apache v2 license
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datasets
import logging
import os
import sys
sys.path.append("/data2/lkk/llama/test_pr/intel-extension-for-transformers")
import transformers
from transformers.modeling_utils import unwrap_model
from dataclasses import dataclass, field
from datasets import load_dataset
from peft import (
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.adaption_prompt import AdaptionPromptConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from typing import Optional, List
import re
import torch
import importlib.util
from transformers.utils.import_utils import is_optimum_available
from data_utils import preprocess_dataset

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


def is_optimum_habana_available():
    return is_optimum_available() and importlib.util.find_spec("optimum.habana") != None


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
        default=384,
        metadata={
            "help": "The maximum total source sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    max_new_tokens: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum generation sequence length when do generation."
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
    eval_dataset_size: int = field(
        default=500, metadata={"help": "Size of validation dataset."}
    )


@dataclass
class FinetuneArguments:
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
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    habana: bool = field(
        default=False,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    lora_all_linear: bool = field(
        default=False,
        metadata={"help": "if True, will add adaptor for all linear for lora finetuning"},
    )
    task: Optional[str] = field(
        default="completion",
        metadata={"help": "task name, different task means different data format.",
            "choices": ["completion", "chat", "summarization"]
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    if not is_optimum_habana_available():
        parser = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, FinetuneArguments)
        )
    else:
        from optimum.habana import GaudiTrainingArguments

        parser = HfArgumentParser(
            (ModelArguments, DataArguments, GaudiTrainingArguments, FinetuneArguments)
        )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, finetune_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            finetune_args,
        ) = parser.parse_args_into_dataclasses()

    if finetune_args.habana:
        if not is_optimum_habana_available():
            raise ImportError(
                "optimum habana is not installed. refer https://github.com/huggingface/optimum-habana"
            )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary
    b16 = training_args.fp16 or training_args.bf16
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
        + f"\ndistributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {b16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True if model_args.trust_remote_code else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        raise ValueError("Please provide value for model_name_or_path or config_name.")
    
    # set use_fast_tokenizer to False for Llama series models
    if "llama" in config.model_type:
        model_args.use_fast_tokenizer = False


    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )


    # Load model
    if model_args.model_name_or_path:
        model_dtype = torch.bfloat16 if training_args.bf16 else None
        if (re.search("mpt", model_args.model_name_or_path, re.IGNORECASE) or
            re.search("neural-chat-7b-v1", model_args.model_name_or_path, re.IGNORECASE)):
            from models.mpt.modeling_mpt import MPTForCausalLM

            model = MPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                trust_remote_code=True if model_args.trust_remote_code else None,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                trust_remote_code=True if model_args.trust_remote_code else None,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
            )
    else:
        raise ValueError(
            "Must provide model_name_or_path to load a pretrained CausalLM model."
        )

    # add special tokens
    if data_args.special_tokens:
        additional_special_tokens = {
            "additional_special_tokens": data_args.special_tokens}
        tokenizer.add_special_tokens(additional_special_tokens)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if re.search("llama", model.config.architectures[0], re.IGNORECASE):
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
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    raw_datasets, preprocess_function = preprocess_dataset(raw_datasets, tokenizer, data_args, finetune_args)
    column_names = list(raw_datasets["train"].features)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )


    if data_args.dataset_concatenation:

        def concatenate_data(dataset, max_seq_length):
            concatenated_dataset = {}
            for column in dataset.features:
                concatenated_data = [
                    item for sample in dataset[column] for item in sample
                ]
                reshaped_data = [
                    concatenated_data[i * max_seq_length : (i + 1) * max_seq_length]
                    for i in range(len(concatenated_data) // max_seq_length)
                ]
                concatenated_dataset[column] = reshaped_data
            return datasets.Dataset.from_dict(concatenated_dataset)

        tokenized_datasets["train"] = concatenate_data(
            tokenized_datasets["train"], data_args.max_seq_length
        )

    if training_args.do_eval:
        if "test" not in tokenized_datasets:
            logger.info('Splitting train dataset in train and validation according to `eval_dataset_size`')
            tokenized_datasets = tokenized_datasets["train"].train_test_split(
                test_size=data_args.eval_dataset_size, shuffle=True, seed=42
            )
        eval_dataset = tokenized_datasets["test"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    logger.info(
        "Using data collator of type {}".format(data_collator.__class__.__name__)
    )

    if training_args.do_train:
        # PEFT settings
        if finetune_args.peft == "lora":
            if finetune_args.lora_all_linear:
                target_modules = find_all_linear_names(model)
            else:
                target_modules = finetune_args.lora_target_modules

            peft_config = LoraConfig(
                r=finetune_args.lora_rank,
                lora_alpha=finetune_args.lora_alpha,
                lora_dropout=finetune_args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        elif finetune_args.peft == "llama_adapter":
            peft_config = AdaptionPromptConfig(
                adapter_layers=finetune_args.adapter_layers,
                adapter_len=finetune_args.adapter_len,
                task_type="CAUSAL_LM",
            )
        elif finetune_args.peft == "ptun":
            peft_config = PromptEncoderConfig(
                num_virtual_tokens=finetune_args.num_virtual_tokens,
                encoder_hidden_size=finetune_args.ptun_hidden_size,
                task_type="CAUSAL_LM",
            )
        elif finetune_args.peft == "prefix":
            peft_config = PrefixTuningConfig(
                num_virtual_tokens=finetune_args.num_virtual_tokens,
                task_type="CAUSAL_LM",
            )
        elif finetune_args.peft == "prompt":
            peft_config = PromptTuningConfig(
                num_virtual_tokens=finetune_args.num_virtual_tokens,
                task_type="CAUSAL_LM",
            )

        model = get_peft_model(model, peft_config)
        if model_dtype == torch.bfloat16:
            model = model.to(model_dtype)
        model.print_trainable_parameters()

        if not finetune_args.habana:
            # Initialize our Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        else:
            from optimum.habana import GaudiConfig, GaudiTrainer

            gaudi_config = GaudiConfig()
            gaudi_config.use_fused_adam = True
            gaudi_config.use_fused_clip_norm = True
            # Initialize our Trainer
            trainer = GaudiTrainer(
                model=model,
                gaudi_config=gaudi_config,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        with training_args.main_process_first(desc="save model"):
            if is_main_process(training_args.local_rank):
                unwrapped_model = unwrap_model(model)
                unwrapped_model.save_pretrained(
                    training_args.output_dir, state_dict=unwrapped_model.state_dict()
                )

        if finetune_args.do_lm_eval and finetune_args.task != "summarization":
            unwrapped_model.eval()
            from intel_extension_for_transformers.evaluation.lm_eval import evaluate
            with training_args.main_process_first(desc="lm_eval"):
                if is_main_process(training_args.local_rank):
                    with torch.no_grad():
                        results = evaluate(
                                model="hf-causal",
                                model_args='pretrained='+model_args.model_name_or_path+\
                                        ',tokenizer='+model_args.model_name_or_path+',dtype=float16',
                                user_model=unwrapped_model,
                                device=unwrapped_model.device.type,
                                batch_size=training_args.per_device_eval_batch_size,
                                tasks=finetune_args.lm_eval_tasks,)
                        logger.info(results)

        if finetune_args.task == "summarization":
            from eval_utils import compute_rouge_metric
            gen_kwargs = {
                    "num_beams": data_args.num_beams,
                    "max_new_tokens": data_args.max_new_tokens,
                    }
            with training_args.main_process_first(desc="summarization eval"):
                if is_main_process(training_args.local_rank):
                    results = compute_rouge_metric(unwrapped_model, tokenizer, eval_dataset,
                            training_args, gen_kwargs)
                    logger.info(results)



if __name__ == "__main__":
    main()
