#!/usr/bin/env python
# coding=utf-8

# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 The HuggingFace Team All rights reserved.
#
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
"""
This script is based on HuggingFace/transformers example: https://github.com/huggingface/transformers/blob/v4.6.1/examples/pytorch/language-modeling/run_mlm.py
Changes made to the script:
 1. Added pruning capabilities
 2. Added model distillation capabilities
 3. Added learning rate rewinding option
 4. Added methods to save all hyper-parameters used
 5. Removed pre-processing code and exported it to dataset_processing.py
"""

import dataset_processing as preprocess
import logging
import math
import numpy as np
import os
import sys
import torch
import transformers

from collections import defaultdict
from dataclasses import dataclass, field
from intel_extension_for_transformers.optimization import (
    AutoDistillationConfig,
    FlashDistillationConfig,
    metrics
)
from intel_extension_for_transformers.optimization.trainer import NLPTrainer as Trainer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import get_scheduler
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from typing import Optional, List


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.10.0")
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_wwm: bool = field(
        default=False,
        metadata={"help": "Use Whole Word Masking DataCollator"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    datasets_name_config: Optional[List[str]] = field(
        default=None, metadata={"help": "The name:config list of datasets to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
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
    data_process_type: str = field(
        default="concatenate",
        metadata={
            "help": f"The preprocess method to use for data preprocessing. Choose from list: {list(preprocess.PROCESS_TYPE.keys())}"
        },
    )
    short_seq_probability: float = field(
        default=0.1,
        metadata={
            "help": "The probability to parse document to shorter sentences than max_seq_length. Defaults to 0.1."
        },
    )
    nsp_probability: float = field(
        default=0.5,
        metadata={
            "help": "The probability to choose a random sentence when creating next sentence prediction examples."
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
        }
    )


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    auto_distillation: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply distillation."},
    )
    teacher_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    teacher_model_name_or_path: str = field(
        default=False,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    flash_distillation_steps: int = field(
        default=500,
        metadata={
            "help": "Steps for each stage in knowledge transfer."
        },
    )
    regular_distillation_steps: int = field(
        default=25000,
        metadata={
            "help": "Steps for each stage in regular distillation."
        },
    )
    max_trials: int = field(
        default=100,
        metadata={
            "help": "Maximum trials for AutoDistillation."
        },
    )

def main():
    if int(os.environ.get("LOCAL_RANK", -1)) != -1 and '--no_cuda' in sys.argv:
        from intel_extension_for_transformers.optimization.utils.utility import distributed_init
        distributed_init()

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, optim_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, optim_args = parser.parse_args_into_dataclasses()

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
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"\ndistributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    model_cls = AutoModelForPreTraining if data_args.data_process_type == 'segment_pair_nsp' else AutoModelForMaskedLM
    if model_args.model_name_or_path:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_cls.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    if not is_main_process(training_args.local_rank):
        torch.distributed.barrier()
    tokenized_datasets = preprocess.data_process(tokenizer, data_args)
    if training_args.local_rank != -1 and is_main_process(training_args.local_rank):
        torch.distributed.barrier()

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = training_args.fp16 and not data_args.pad_to_max_length
    if model_args.use_wwm:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    logger.info("Using data collator of type {}".format(data_collator.__class__.__name__))

    if optim_args.auto_distillation:
        if optim_args.teacher_config_name:
            teacher_config = AutoConfig.from_pretrained(optim_args.teacher_config_name, \
                                                        **config_kwargs)
        else:
            teacher_config = AutoConfig.from_pretrained(optim_args.teacher_model_name_or_path, \
                                                        **config_kwargs)
        teacher_tokenizer = AutoTokenizer.from_pretrained(optim_args.teacher_model_name_or_path, \
                                                          **tokenizer_kwargs)
        assert teacher_tokenizer.vocab == tokenizer.vocab, \
                'teacher model and student model should have same tokenizer.'
        teacher_model = model_cls.from_pretrained(
            optim_args.teacher_model_name_or_path,
            from_tf=bool(".ckpt" in optim_args.teacher_model_name_or_path),
            config=teacher_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        teacher_model.to(training_args.device)
        
        para_counter = lambda model:sum(p.numel() for p in model.parameters())
        logger.info("***** Number of teacher model parameters: {:.2f}M *****".format(\
                    para_counter(teacher_model)/10**6))
        logger.info("***** Number of student model parameters: {:.2f}M *****".format(\
                    para_counter(model)/10**6))

    metric = None
    if data_args.data_process_type == 'segment_pair_nsp':
        def metric(eval_prediction):
            def accuracy(logits, labels):
                mask = labels != -100
                preds = np.argmax(logits[mask], axis=-1)
                return (preds == labels[mask]).mean()

            def loss(logits, labels):
                mask = labels != -100
                logits = torch.tensor(logits[mask])
                labels = torch.tensor(labels[mask])
                return torch.nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), 
                                                   labels.view(-1)).item()

            mlm_acc = accuracy(eval_prediction.predictions[0], eval_prediction.label_ids[0])
            nsp_acc = accuracy(eval_prediction.predictions[1], eval_prediction.label_ids[1])
            mlm_loss = loss(eval_prediction.predictions[0], eval_prediction.label_ids[0])
            nsp_loss = loss(eval_prediction.predictions[1], eval_prediction.label_ids[1])
            perplexity = np.exp(mlm_loss).item()
            return {
                "mlm_acc": mlm_acc,
                "mlm_loss": mlm_loss,
                "perplexity": perplexity,
                "nsp_acc": nsp_acc,
                "nsp_loss": nsp_loss,
                "accuracy": (mlm_acc + nsp_acc) / 2,
            }
        model.config.keys_to_ignore_at_inference = ['attentions']
        training_args.label_names = ['labels', 'next_sentence_label']
        if optim_args.auto_distillation:
            teacher_model.config.keys_to_ignore_at_inference = ['attentions']

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric,
    )

    # Auto Distillation
    if optim_args.auto_distillation:
        if 'mobilebert' in model_args.config_name:
            # for MobileBERT
            stages = 24
            autodistillation_config = \
                AutoDistillationConfig(
                    search_space={
                        'hidden_size': [128, 246, 384, 512],
                        'intra_bottleneck_size': [64, 96, 128, 160],
                        'num_attention_heads': [1, 2, 4, 8],
                        'intermediate_size': [384, 512, 640],
                        'num_feedforward_networks': [2, 4, 6]
                        },
                    max_trials=optim_args.max_trials,
                    metrics=[
                        metrics.Metric(name="eval_loss", greater_is_better=False),
                        metrics.Metric(name="latency", greater_is_better=False),
                    ],
                    knowledge_transfer=FlashDistillationConfig(
                        block_names=['mobilebert.encoder.layer.{}'.format(i) for i in range(stages)],
                        layer_mappings_for_knowledge_transfer=[
                            [
                                [
                                    ('mobilebert.encoder.layer.{}.attention.self'.format(i), '1'),
                                    ('bert.encoder.layer.{}.attention.self'.format(i), '1')
                                ],
                                [
                                    ('mobilebert.encoder.layer.{}.output'.format(i),),
                                    ('bert.encoder.layer.{}.output'.format(i),)
                                ]
                            ] for i in range(stages)
                        ],
                        loss_types=[['KL', 'MSE'] for i in range(stages)],
                        loss_weights=[[0.5, 0.5] for i in range(stages)],
                        train_steps=[optim_args.flash_distillation_steps for i in range(stages)]
                    ),
                    regular_distillation=FlashDistillationConfig(
                        layer_mappings_for_knowledge_transfer=[
                            [[('cls', '0')]]
                        ],
                        loss_types=[['KL']],
                        add_origin_loss=[True],
                        train_steps=[optim_args.regular_distillation_steps]
                    ),
            )
        elif 'bert-tiny' in model_args.config_name:
            # for BERT-Tiny
            autodistillation_config = AutoDistillationConfig(
                    search_space={
                        'hidden_size': [64, 128, 256, 384],
                        'num_attention_heads': [1, 2, 4, 8, 16],
                        'intermediate_size': [128, 256, 384, 512, 640],
                    },
                    max_trials=optim_args.max_trials,
                    metrics=[
                        metrics.Metric(name="eval_loss", greater_is_better=False),
                        metrics.Metric(name="latency", greater_is_better=False),
                    ],
                    knowledge_transfer=FlashDistillationConfig(
                        block_names=['bert.encoder.layer.0', 'bert.encoder.layer.1'],
                        layer_mappings_for_knowledge_transfer=[
                                [
                                    [
                                        ('bert.encoder.layer.0.attention.self', '1')
                                    ],
                                    [
                                        ('bert.encoder.layer.0.output',)
                                    ]
                                ],
                                [
                                    [
                                        ('bert.encoder.layer.1.attention.self', '1'),
                                        ('bert.encoder.layer.11.attention.self', '1')
                                    ],
                                    [
                                        ('bert.encoder.layer.1.output',),
                                        ('bert.encoder.layer.11.output',)
                                    ]
                                ]
                            ],
                        loss_types=[['KL', 'MSE'], ['KL', 'MSE']],
                        loss_weights=[[0.5, 0.5], [0.5, 0.5]],
                        train_steps=[optim_args.flash_distillation_steps] * 2
                    ),
                    regular_distillation=FlashDistillationConfig(
                        layer_mappings_for_knowledge_transfer=[
                            [[('cls', '0')]]
                        ],
                        loss_types=[['KL']],
                        add_origin_loss=[True],
                        train_steps=[optim_args.regular_distillation_steps]
                    ),
            )
        best_model_archs = trainer.autodistillation(
            autodistillation_config,
            teacher_model,
            model_cls=model_cls
        )
        print("Best model architectures obtained by AutoDistillation are as follow.")
        print(best_model_archs)
    else:
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics_result = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics_result["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics_result)
            trainer.save_metrics("train", metrics_result)
            trainer.save_state()
            try:
                torch.save([vars(a) for a in [training_args, data_args, model_args]], os.path.join(training_args.output_dir, "args.bin"))
            except:
                logger.info("Failed to save arguments")

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics_result = trainer.evaluate()

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics_result["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            if "eval_nsp_loss" in metrics_result:
                try:
                    perplexity = math.exp(metrics_result["eval_loss"] - metrics_result["eval_nsp_loss"])
                except:
                    logger.warning("Perplexity computation failed")
                    perplexity = math.exp(metrics_result["eval_loss"])
            else:
                    perplexity = math.exp(metrics_result["eval_loss"])
            metrics_result["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics_result)
            trainer.save_metrics("eval", metrics_result)

        if training_args.push_to_hub:
            kwargs = {"finetuned_from": model_args.model_name_or_path, "tags": "fill-mask"}
            if data_args.dataset_name is not None:
                kwargs["dataset_tags"] = data_args.dataset_name
                if data_args.dataset_config_name is not None:
                    kwargs["dataset_args"] = data_args.dataset_config_name
                    kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
                else:
                    kwargs["dataset"] = data_args.dataset_name

            trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
