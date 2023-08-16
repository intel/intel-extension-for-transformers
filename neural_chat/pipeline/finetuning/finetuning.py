#!/usr/bin/env python
# coding=utf-8

# Apache v2 license
# Copyright (C) 2023 Intel Corporation
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
import transformers
from transformers.modeling_utils import unwrap_model
from datasets import load_dataset
from peft import (
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    PeftModel,
    PeftConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.adaption_prompt import AdaptionPromptConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
import re
import numpy as np
import evaluate
import torch
import importlib.util
from transformers.utils.import_utils import is_optimum_available
from .data_utils import preprocess_dataset, ALPACA_PROMPT_DICT
from neural_chat.config import FinetuningConfig


def is_optimum_habana_available():
    return is_optimum_available() and importlib.util.find_spec("optimum.habana") != None


class Finetuning:
    def __init__(self, finetuning_config: FinetuningConfig):
        self.model_args, self.data_args, self.training_args, self.finetune_args = (
            finetuning_config.model_args, 
            finetuning_config.data_args, 
            finetuning_config.training_args,
            finetuning_config.finetune_args
        )
        if finetuning_config.finetune_args.device == "cpu":
            finetuning_config.training_args.no_cuda = True
            Arguments = type(finetuning_config.training_args)
            training_args = {
                k: getattr(finetuning_config.training_args, k) \
                    for k in Arguments.__dataclass_fields__.keys() if Arguments.__dataclass_fields__[k].init
            }
            self.training_args = Arguments(**training_args)

        os.environ["WANDB_DISABLED"] = "true"

        self.setup_logger()

        # Log on each process the small summary
        b16 = self.training_args.fp16 or self.training_args.bf16
        self.logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}"
            + f"\ndistributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {b16}"
        )
        self.logger.info(f"Training/evaluation parameters {self.training_args}")

        # Set seed before initializing model.
        set_seed(self.training_args.seed)

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        if self.training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = self.training_args.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    def load_dataset(self, data_args, model_args, training_args):
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
                streaming=data_args.streaming,
            )

            if "validation" not in raw_datasets.keys() and training_args.do_eval:
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

            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys() and training_args.do_eval:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    **dataset_args,
                )
        return raw_datasets

    def load_model_config(self, model_args):
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
        return config

    def load_tokenizer(self, model_args):
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
        return tokenizer

    def finetune(self):
        config = self.load_model_config(self.model_args)
        if config.architectures[0].endswith("ForCausalLM"):
            self.finetune_clm()
        elif config.architectures[0].endswith("ForConditionalGeneration"):
            self.finetune_seq2seq()
        else:
            raise NotImplementedError(
                "Unsupported architecture {}, only support CausalLM (CLM) and ConditionalGeneration (Seq2seq) now.".format(
                    config.architectures[0]
                )
            )

    def finetune_clm(self):
        model_args, data_args, training_args, finetune_args = \
            self.model_args, self.data_args, self.training_args, self.finetune_args

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

        if finetune_args.device == 'habana':
            if not is_optimum_habana_available():
                raise ImportError(
                    "optimum habana is not installed. refer https://github.com/huggingface/optimum-habana"
                )

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = self.load_model_config(model_args)
        
        # set use_fast_tokenizer to False for Llama series models
        if "llama" in config.model_type:
            model_args.use_fast_tokenizer = False

        tokenizer = self.load_tokenizer(model_args)

        raw_datasets = self.load_dataset(data_args, model_args, training_args)

        # Load model
        if model_args.model_name_or_path:
            model_dtype = torch.bfloat16 if training_args.bf16 else None
            if re.search("mpt", model_args.model_name_or_path, re.IGNORECASE):
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

        with training_args.main_process_first(desc="dataset map pre-processing"):
            tokenized_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
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

            tokenized_datasets_ = tokenized_datasets["train"].remove_columns(
                ["prompt_sources", "prompt_targets"]
            )
            tokenized_datasets["train"] = concatenate_data(
                tokenized_datasets_, data_args.max_seq_length
            )

        if training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = tokenized_datasets["train"]
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            if "test" not in tokenized_datasets:
                raise ValueError("--do_eval requires a test dataset")
            eval_dataset = tokenized_datasets["test"]
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        self.logger.info(
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

            if finetune_args.device != 'habana':
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

    def finetune_seq2seq(self):
        model_args, data_args, training_args, finetune_args = \
            self.model_args, self.data_args, self.training_args, self.finetune_args

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                self.logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        raw_datasets = self.load_dataset(data_args, model_args, training_args)

        tokenizer = self.load_tokenizer(model_args)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)

        # dataset preprocessing
        def prepare_features(examples):
            instructions = [q.strip() for q in examples["instruction"]]
            inputs = [q.strip() for q in examples["input"]]
            responses = [q.strip() for q in examples["output"]]
            examples["input_ids"] = []
            examples["labels"] = []
            for instruction, input, response in zip(instructions, inputs, responses):
                if input == "":
                    prompt = ALPACA_PROMPT_DICT["prompt_without_input"].format(instruction=instruction)
                else:
                    prompt = ALPACA_PROMPT_DICT["prompt_with_input"].format(instruction=instruction, input=input)

                history = tokenizer(prompt, max_length=data_args.max_source_length,
                        truncation=True, add_special_tokens=False)
                gt_resp = tokenizer(response, max_length=data_args.max_target_length, truncation=True)
                input_ids = history.input_ids
                labels = gt_resp.input_ids
                examples["input_ids"].append(input_ids)
                examples["labels"].append(labels)

            return examples

        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                # We will select sample from whole data if argument is specified
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
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
            if data_args.max_train_samples is not None:
                # Number of samples might increase during Feature Creation, We select only specified max samples
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        # Metric
        metric = evaluate.load("rouge")

        import nltk
        nltk.download("punkt", quiet=True)

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        # Define compute metrics function
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result


        if training_args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_examples = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                # We will select sample from whole data
                max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
                eval_examples = eval_examples.select(range(max_eval_samples))
            # Validation Feature Creation
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_examples.map(
                    prepare_features,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
            if data_args.max_eval_samples is not None:
                # During Feature creation dataset samples might increase, we will select required samples again
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)
        
        if training_args.do_train:
            # download model & vocab.
            config = self.load_model_config(model_args)

            # Load model
            if model_args.model_name_or_path:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                model.resize_token_embeddings(len(tokenizer))
            else:
                raise ValueError("Must provide model_name_or_path to load a pretrained Seq2SeqLM model.")

            # PEFT settings
            if finetune_args.peft == "lora":
                peft_config = LoraConfig(
                    r=finetune_args.lora_rank,
                    lora_alpha=finetune_args.lora_alpha,
                    lora_dropout=finetune_args.lora_dropout,
                    target_modules=finetune_args.lora_target_modules,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM,
                )

            # model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        if training_args.do_eval and not training_args.do_train:
            config = PeftConfig.from_pretrained(model_args.model_name_or_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path)

        # ignore tokenizer pad token in the loss
        label_pad_token_id = -100
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8)

        # Create Trainer instance
        trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
                )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            self.logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)