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
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.
"""
This script is based on HuggingFace/transformers example: https://github.com/huggingface/transformers/blob/v4.6.1/examples/pytorch/question-answering/run_qa.py
Changes made to the script:
 1. Added pruning capabilities
 2. Added model distillation capabilities
 3. Added learning rate rewinding option
 4. Added methods to save all hyper-parameters used
 5. Added quantization capabilities
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from tqdm.auto import tqdm
import math

import torch
import datasets
from datasets import load_dataset, load_metric

import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_scheduler,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    SchedulerType
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.file_utils import get_full_repo_name

from utils_qa import postprocess_qa_predictions

from huggingface_hub import Repository

from functools import partial
from accelerate import Accelerator
from torch.utils.data import DataLoader
import argparse
from accelerate.logging import get_logger
import numpy as np
import utils_qa
import json
from neural_compressor.training import Pruning, prepare_compression
from neural_compressor.training import WeightPruningConfig

os.environ["WANDB_DISABLED"] = "true"
os.environ["HTTP_PROXY"] = ""

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# (['loss', 'start_logits', 'end_logits'])
# batch(['attention_mask', 'end_positions', 'input_ids', 'start_positions', 'token_type_ids']
def get_loss_one_logit(student_logit, teacher_logit):
    t = 2.0
    from torch.nn import functional as F
    return F.kl_div(
        input=F.log_softmax(student_logit / t, dim=-1),
        target=F.softmax(teacher_logit / t, dim=-1),
        reduction="batchmean"
    ) * (t ** 2)

def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    """
    Save results while prefixing metric names.
    Args:
        results: (:obj:`dict`):
            A dictionary of results.
        output_dir: (:obj:`str`):
            An output directory.
        file_name: (:obj:`str`, `optional`, defaults to :obj:`all_results.json`):
            An output file name.
        metric_key_prefix: (:obj:`str`, `optional`, defaults to :obj:`eval`):
            A metric name prefix.
    """
    # Prefix all keys with metric_key_prefix + '_'
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int, default=10,
        help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="To do prediction on the question answering model"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--distill_loss_weight",
        type=float,
        default=0.0,
        help="distiller loss weight"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--warm_epochs",
        type=int,
        default=0,
        help="Number of epochs the network not be purned"
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--cooldown_epochs",
        type=int, default=0,
        help="Cooling epochs after pruning."
    )
    parser.add_argument(
        "--do_prune", action="store_true",
        help="Whether or not to prune the model"
    )
    parser.add_argument(
        "--pruning_scope",
        type=str, default="global",
        help="pruning scope, we support global and local."
    )
    parser.add_argument(
        "--pruning_pattern",
        type=str, default="4x1",
        help="pruning pattern type, we support NxM and N:M."
    )
    parser.add_argument(
        "--target_sparsity",
        type=float, default=0.8,
        help="Target sparsity of the model."
    )
    parser.add_argument(
        "--pruning_frequency",
        type=int, default=-1,
        help="Sparse step frequency for iterative pruning, default to a quarter of pruning steps."
    )

    parser.add_argument(
        "--keep_conf", action="store_true",
        help="Whether or not to keep the prune config infos"
    )
    parser.add_argument(
        "--pruning_config",
        type=str,
        help="pruning_config"
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to directory to store the pretrained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )

    parser.add_argument(
        "--use_auth_token",
        type=bool,
        default=False,
        help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).",
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.",
    )

    args = parser.parse_args()

    # Sanity checks
    if (
            args.dataset_name is None
            and args.train_file is None
            and args.validation_file is None
            and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def main():

    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_qa_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    
    '''
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    '''
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    script_path = os.path.split(os.path.abspath(__file__))[0]

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]

        if args.validation_file is not None:
            data_files["dev"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        # datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)
        raw_datasets = load_dataset(os.path.join(script_path, "squad.py"), data_files=data_files, cache_dir=args.cache_dir)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    # local py module
    from modeling_longformer import LongformerForQuestionAnswering
    model_class = LongformerForQuestionAnswering

    if args.distill_loss_weight > 0:
        teacher_path = args.teacher_model_name_or_path 
        if teacher_path is None:
            teacher_path = args.model_name_or_path
        teacher_model = model_class.from_pretrained(
            teacher_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class.from_config(config)

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if args.do_train:
        column_names = raw_datasets["train"].column_names
    elif args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    # pad_on_right = tokenizer.padding_side == "right"
    # max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    max_seq_length = args.max_seq_length

    # preprocess context and answers
    def preprocess_context(examples):
        new_examples = {}

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        def pre_tokenize(p):
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in p:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            return ' '.join(doc_tokens), char_to_word_offset

        new_examples[context_column_name] = []
        new_examples["answer_spans"] = []
        for i, p in enumerate(examples[context_column_name]):
            tokenized_p, char_to_word_offset = pre_tokenize(p)
            new_examples[context_column_name].append(tokenized_p)

            answer_spans = []
            for orig_answer_text, answer_offset in zip(examples[answer_column_name][i]['text'], examples[answer_column_name][i]['answer_start']):
                answer_length = len(orig_answer_text)
                try:
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    token_ids = tokenizer.encode(orig_answer_text)
                except RuntimeError:
                    logger.info(f'Reading example {idx} failed')
                    start_position = 0
                    end_position = 0
                answer_spans.append({'start': start_position, 'end': end_position,
                    'text': orig_answer_text, 'token_ids': token_ids})
            new_examples["answer_spans"].append(answer_spans)

        for key in examples:
            if key != context_column_name:
                new_examples[key] = examples[key]
        return new_examples

    # preprocessing
    def prepare_features(examples, max_question_len=55, max_doc_len=4096, max_num_answers=64, ignore_seq_with_no_answers=False, mode="eval"):

        tokenized_examples = {}
        tokenized_examples["input_ids"] = []
        tokenized_examples["attention_mask"] = []
        if mode == "train":
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
        elif mode == "eval":
            tokenized_examples["example_id"] = []
        else:
            raise NotImplementedError("not implemented yet.")

        # not use for roberta
        #tokenized_examples["token_type_ids"] = []

        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        for example_index in range(len(examples[question_column_name])):
            question_text = examples[question_column_name][example_index]
            query_tokens = tokenizer.tokenize(question_text)
            query_tokens = query_tokens[:max_question_len]
            doc_tokens = examples[context_column_name][example_index].split(" ")
            answer_spans = examples["answer_spans"][example_index]
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(f'. {token}')[1:] if i > 0 else tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
            all_doc_tokens = all_doc_tokens[:max_doc_len]
            # The -3 accounts for <s>, </s> and </s>
            max_tokens_per_doc_slice = max_seq_length - len(query_tokens) - 3
            assert max_tokens_per_doc_slice > 0

            if args.doc_stride < 0:
                # negative doc_stride indicates no sliding window, but using first slice
                args.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once

            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []
            start_positions_list = []
            end_positions_list = []
            answer_token_ids_list = []

            for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - args.doc_stride):
                slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))
                doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token] \
                        + doc_slice_tokens + [tokenizer.sep_token]

                # but don't use for roberta
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 1)
                assert len(segment_ids) == len(tokens)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                #if data_args.pad_to_max_length:  # no need to pad if document is not strided
                if False:
                    # Zero-pad up to the sequence length.
                    padding_len = max_seq_length - len(input_ids)
                    input_ids.extend([tokenizer.pad_token_id] * padding_len)
                    input_mask.extend([0] * padding_len)
                    segment_ids.extend([0] * padding_len)

                    assert len(input_ids) == max_seq_length
                    assert len(input_mask) == max_seq_length
                    assert len(segment_ids) == max_seq_length

                doc_offset = len(query_tokens) + 2 - slice_start

                start_positions = []
                end_positions = []
                answer_token_ids = []
                for answer_span in answer_spans:
                    start_position = answer_span['start']
                    end_position = answer_span['end']
                    tok_start_position_in_doc = orig_to_tok_index[start_position]
                    not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                    tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
                    if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                        # this answer is outside the current slice
                        continue

                    start_positions.append(tok_start_position_in_doc + doc_offset)
                    end_positions.append(tok_end_position_in_doc + doc_offset)
                    answer_token_ids.append(answer_span['token_ids'])

                assert len(start_positions) == len(end_positions)
                if ignore_seq_with_no_answers and len(start_positions) == 0:
                    continue

                # answers from start_positions and end_positions if > self.max_num_answers
                start_positions = start_positions[:max_num_answers]
                end_positions = end_positions[:max_num_answers]
                answer_token_ids = answer_token_ids[:max_num_answers]

                # -1 padding up to self.max_num_answers
                # -1 means empty answer in last token, while normal squad in [CLS] token
                padding_len = max_num_answers - len(start_positions)
                start_positions.extend([-1] * padding_len)
                end_positions.extend([-1] * padding_len)
                answer_token_ids.extend([[]] * padding_len)

                # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
                found_start_positions = set()
                found_end_positions = set()
                found_answer_token_ids = set()
                for i, (start_position, end_position, answer_tokens) in enumerate(
                        zip(start_positions, end_positions, answer_token_ids)
                        ):
                    if start_position in found_start_positions:
                        start_positions[i] = -1
                    if end_position in found_end_positions:
                        end_positions[i] = -1
                    answer_tokens_as_str = ','.join([str(x) for x in answer_tokens])
                    if answer_tokens_as_str in found_answer_token_ids:
                        answer_token_ids[i] = []

                    found_start_positions.add(start_position)
                    found_end_positions.add(end_position)
                    found_answer_token_ids.add(answer_tokens_as_str)

                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)
                start_positions_list.append(start_positions)
                end_positions_list.append(end_positions)
                answer_token_ids_list.append(answer_token_ids)

            # pad answers in answer_token_ids_list to the longest answer
            max_answer_len = max([len(item) for sublist in answer_token_ids_list for item in sublist])  # flat list
            if max_answer_len == 0:
                max_answer_len = 2
            for answers_of_one_slice in answer_token_ids_list:
                for answer_tokens in answers_of_one_slice:
                    if len(answer_tokens) == 0:
                        # TODO: <s></s><pad><pad><pad> or <pad><pad><pad><pad><pad> ?
                        padding_len = max_answer_len - len(answer_tokens) - 2
                        answer_tokens.extend([tokenizer.bos_token_id, tokenizer.eos_token_id] +
                                                 ([tokenizer.pad_token_id] * padding_len))
                    else:
                        padding_len = max_answer_len - len(answer_tokens)
                        answer_tokens.extend([tokenizer.pad_token_id] * padding_len)


            tokenized_examples["input_ids"].extend(input_ids_list)
            tokenized_examples["attention_mask"].extend(input_mask_list)

            if mode == "train":
                # only one answer used for training
                #tokenized_examples["start_positions"].extend([each[0] for each in start_positions_list])
                #tokenized_examples["end_positions"].extend([each[0] for each in end_positions_list])
                tokenized_examples["start_positions"].append(start_positions_list[0])
                tokenized_examples["end_positions"].append(end_positions_list[0])
            elif mode == "eval":
                tokenized_examples["example_id"].append(examples["id"][example_index])

        return tokenized_examples

    prepare_train_features = partial(prepare_features, mode="train")
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(args.max_train_samples))
        with accelerator.main_process_first():
            # preprocess
            train_dataset = train_dataset.map(
                    preprocess_context,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    )

            # Create train feature from dataset
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names + ["answer_spans"],
                load_from_cache_file=not args.overwrite_cache,
            )
        if args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(args.max_train_samples))

    prepare_validation_features = partial(prepare_features, mode="eval")

    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(args.max_eval_samples))
        with accelerator.main_process_first():
            # preprocess
            eval_examples = eval_examples.map(
                    preprocess_context,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    )
            # Validation Feature Creation
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
            )

        if args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))


    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "answer_spans"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            tokenizer=tokenizer,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name], "aliases": ex["aliases"]} for ex in examples]

        return EvalPrediction(predictions=predictions, label_ids=references)

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step: step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    no_decay_outputs = ["bias", "LayerNorm.weight", "qa_outputs"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.do_prune:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.9])
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.distill_loss_weight > 0:
        teacher_model, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            teacher_model, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        teacher_model.eval()
    else:
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("qa_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # Pruning preparation
    num_iterations = len(train_dataset) / total_batch_size
    num_warm = int(args.warm_epochs * num_iterations) + args.num_warmup_steps
    total_iterations = int(num_iterations * (args.num_train_epochs - args.cooldown_epochs))
    frequency = int((total_iterations - num_warm + 1) / 40) if args.pruning_frequency == -1 \
                                                           else args.pruning_frequency

    pruning_start = num_warm
    pruning_end = total_iterations
    if not args.do_prune:
        pruning_start = num_iterations * args.num_train_epochs + 1
        pruning_end = pruning_start

    pruning_configs=[
        {
            "pruning_type": "snip_momentum",
            "pruning_scope": "global",
            "sparsity_decay_type": "exp",
            "excluded_op_names": ["qa_outputs", "pooler", ".*embeddings*"],
            "pruning_op_types": ["Linear"],
            "max_sparsity_ratio_per_op": 0.98
        }
    ]

    configs = WeightPruningConfig(
        pruning_configs,
        pruning_scope=args.pruning_scope,
        target_sparsity=args.target_sparsity,
        pattern=args.pruning_pattern,
        pruning_frequency=frequency,
        start_step=pruning_start,
        end_step=pruning_end
    )

    compression_manager = prepare_compression(model=model, confs=configs)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if epoch >= args.warm_epochs:
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                compression_manager.callbacks.on_step_begin(step)

                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                if args.distill_loss_weight > 0:
                    distill_loss_weight = args.distill_loss_weight
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**batch)
                    loss = (distill_loss_weight) / 2 * get_loss_one_logit(outputs['start_logits'],
                                                                        teacher_outputs['start_logits']) \
                        + (distill_loss_weight) / 2 * get_loss_one_logit(outputs['end_logits'],
                                                                        teacher_outputs['end_logits'])
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                    
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    compression_manager.callbacks.on_before_optimizer_step()
                    optimizer.step()
                    compression_manager.callbacks.on_after_optimizer_step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1


                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break
        else:
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        # eval each epoch
        logger.info(f"***** Running Evaluation*****")
        all_start_logits = []
        all_end_logits = []

        # pruner.on_before_eval()
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        # pruner.on_after_eval()

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        eval_preds = post_processing_function(eval_examples, eval_dataset, outputs_numpy)

        metrics = utils_qa.evaluate_triviaqa(eval_preds.label_ids, eval_preds.predictions)
        logger.info(metrics)


    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model.model)
        unwrapped_model.save_pretrained(
            args.output_dir + f"eph{args.num_train_epochs}_lr{args.learning_rate}_bs{total_batch_size}",
            is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            logger.info(json.dumps(metrics, indent=4))
            save_prefixed_metrics(metrics, args.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
