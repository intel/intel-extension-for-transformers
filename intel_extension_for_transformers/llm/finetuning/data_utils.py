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

import copy
import datasets
import re
from itertools import chain
from intel_extension_for_transformers.neural_chat.prompts.prompt import PromptTemplate

IGNORE_INDEX = -100

ALPACA_PROMPT_DICT = {
    "prompt_with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > 0 and len(sequences) > 0:
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]

    return sequences

class CompletionDataPreprocess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()
        if "alpaca" in self.dataset_name:
            self.prompt_template = [
                PromptTemplate("alpaca_without_input"),
                PromptTemplate("alpaca_with_input")
            ]
            self.key_role_map = [
                [('instruction', 0), ('output', 1)],
                [('instruction', 0), ('input', 1), ('output', 2)]
            ]
        elif "stack-exchange-instruction" in self.dataset_name:
            self.prompt_template = PromptTemplate("question_answer")
            self.key_role_map = [('question', 0), ('response', 1)]
        else:
            raise NotImplementedError(
                f"Unsupported dataset {dataset_name}, "
                "only supports stack-exchange-instruction and Alpaca liked dataset now."
            )


    def create_data(self, examples):
        prompts = {}
        prompts["source"] = []
        prompts["target"] = []
        for example in examples:
            prompt_template = self.prompt_template
            key_role_map = self.key_role_map
            if "alpaca" in self.dataset_name:
                if "input" in example and example["input"]:
                    prompt_template = self.prompt_template[1]
                    key_role_map = self.key_role_map[1]
                else:
                    prompt_template = self.prompt_template[0]
                    key_role_map = self.key_role_map[0]

            for idx, (key, role) in enumerate(key_role_map):
                message = example[key]
                if idx == len(key_role_map)-1:
                    message = ""
                prompt_template.append_message(prompt_template.roles[role], message)
            source = prompt_template.get_prompt()
            prompts["source"].append(source)
            prompts["target"].append(example[key_role_map[-1][0]])
            prompt_template.clear_messages()
        return prompts

    @staticmethod
    def tokenize_func(tokenizer, data_args, finetune_args):
        def tokenize(prompt, add_eos_token=True):
            results = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=data_args.max_seq_length,
                    padding=False,
                    return_tensors=None,)
            for i in range(len(results["input_ids"])):
                if (results["input_ids"][i][-1] != tokenizer.eos_token_id \
                        and len(results["input_ids"][i]) < data_args.max_seq_length \
                        and add_eos_token \
                        ):
                    results["input_ids"][i].append(tokenizer.eos_token_id)
                    results["attention_mask"][i].append(1)
            results["labels"] = copy.deepcopy(results["input_ids"])
            results["input_id_len"] = [len(result) for result in results["input_ids"]]
            return results

        def preprocess_function(examples):
            st = [s + t for s, t in zip(examples["prompt_sources"], examples["prompt_targets"])]
            examples_tokenized = tokenize(st)
            input_ids = examples_tokenized["input_ids"]
            labels = examples_tokenized["labels"]
            if not finetune_args.train_on_inputs:
                sources_tokenized = tokenize(examples["prompt_sources"], add_eos_token=False)
                for label, source_len in zip(labels, sources_tokenized["input_id_len"]):
                    label[:source_len] = [IGNORE_INDEX] * source_len
            return dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=examples_tokenized["attention_mask"],
                    )

        return preprocess_function

class IntelDpoDataPreprocess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()

    def create_data(self, examples):
        prompts = {}
        prompts["source"] = []
        prompts["target"] = []
        for example in examples:
            prompts["source"].append(example["system"] + example["question"])
            prompts["target"].append(example["chosen"])
        return prompts

    @staticmethod
    def tokenize_func(tokenizer, data_args, finetune_args):
        def tokenize(prompt, add_eos_token=True):
            results = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=data_args.max_seq_length,
                    padding=False,
                    return_tensors=None,)
            for i in range(len(results["input_ids"])):
                if (results["input_ids"][i][-1] != tokenizer.eos_token_id \
                        and len(results["input_ids"][i]) < data_args.max_seq_length \
                        and add_eos_token \
                        ):
                    results["input_ids"][i].append(tokenizer.eos_token_id)
                    results["attention_mask"][i].append(1)
            results["labels"] = copy.deepcopy(results["input_ids"])
            results["input_id_len"] = [len(result) for result in results["input_ids"]]
            return results

        def preprocess_function(examples):
            st = [s + t for s, t in zip(examples["prompt_sources"], examples["prompt_targets"])]
            examples_tokenized = tokenize(st)
            input_ids = examples_tokenized["input_ids"]
            labels = examples_tokenized["labels"]
            if not finetune_args.train_on_inputs:
                sources_tokenized = tokenize(examples["prompt_sources"], add_eos_token=False)
                for label, source_len in zip(labels, sources_tokenized["input_id_len"]):
                    label[:source_len] = [IGNORE_INDEX] * source_len
            return dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=examples_tokenized["attention_mask"],
                    )

        return preprocess_function


class ChatDataPreprocess:
    base_template = """### System:
    - You are a helpful assistant chatbot trained by Intel.
    - You answer questions.
    - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.{eos_token}\n"""  # pylint: disable=C0301

    def __init__(self, eos_token):
        self.prompt_template = self.base_template.format_map({"eos_token": eos_token})
        self.user = "### User:\n"
        self.assistant = "### Assistant:\n"
        self.end = eos_token

    def create_data(self, examples):
        prompts = {}
        prompts["prompt_sources"] = []
        prompts["prompt_targets"] = []

        for conv in examples:
            conv = conv["messages"]
            prompt = self.prompt_template

            for j in range(0, len(conv) - 1, 2):
                u = conv[j]["content"]
                ass = conv[j+1]["content"]
                prompt = prompt + self.user + u + self.end + '\n' + self.assistant
                response = ass + self.end
                prompts["prompt_sources"].append(prompt)
                prompts["prompt_targets"].append(response)

                prompt += response + '\n'
        return prompts

    def tokenize_func(self, tokenizer, data_args, finetune_args):

        # special tokens
        assistant_tokens = tokenizer.tokenize(self.assistant)

        def preprocess_function(examples):

            instructions = [q.strip() for q in examples["prompt_sources"]]
            responses = [q.strip() for q in examples["prompt_targets"]]

            examples["input_ids"] = []
            examples["labels"] = []
            examples["attention_mask"] = []

            for instruction, response in zip(instructions, responses):
                header = re.findall(r"### System.*?{}".format(self.end), instruction, re.DOTALL)[0]
                convs = re.findall(r"### User.*?{0}|### Assistant.*?{0}".format(self.end), instruction, re.DOTALL)
                convs_tokens = [
                    tokenizer.tokenize(conv) + tokenizer.tokenize("\n")
                    for conv in convs
                ]
                header_tokens = tokenizer.tokenize(header) + tokenizer.tokenize("\n")

                max_input = data_args.max_source_length - len(header_tokens) - len(assistant_tokens)

                truncated_convs = truncate_sequences(convs_tokens,
                        max_input)

                if len(truncated_convs) == 0:
                    truncated_convs = [convs_tokens[-1][:max_input - 3] + convs_tokens[-1][-3:]]

                prompt_tokens = [header_tokens] + truncated_convs + [assistant_tokens]
                prompt_ids = [tokenizer.convert_tokens_to_ids(prompt_token) for prompt_token in prompt_tokens]
                prompt_ids = list(chain(*prompt_ids))

                resp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response.strip()))
                # keep last and eos_id
                max_resp = data_args.max_seq_length - len(prompt_ids) - 1
                if len(resp_ids) > max_resp:
                    resp_ids = resp_ids[:max_resp - 1] + resp_ids[-1:]

                input_ids = prompt_ids + resp_ids  + [tokenizer.eos_token_id]
                if not finetune_args.train_on_inputs:
                    labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids + [tokenizer.eos_token_id]
                else:
                    labels = prompt_ids + resp_ids + [tokenizer.eos_token_id]

                # padding
                input_len = len(input_ids)
                pad_len = data_args.max_seq_length - input_len
                input_ids = input_ids + [tokenizer.eos_token_id] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len
                attention_mask = [1] * input_len + [0] * pad_len

                assert len(input_ids) == data_args.max_seq_length
                assert len(prompt_ids) <= data_args.max_source_length
                assert len(labels) == len(input_ids) == len(attention_mask)

                examples["input_ids"].append(input_ids)
                examples["labels"].append(labels)
                examples["attention_mask"].append(attention_mask)

            return examples

        return preprocess_function

class SlimOrcaDataPreprocess(ChatDataPreprocess):
    def __init__(self, eos_token):
        self.system = "### System:\n"
        self.default_system = "You are a helpful, respectful and honest assistant."
        self.user = "### User:\n"
        self.assistant = "### Assistant:\n"
        self.end = eos_token

    def create_data(self, examples):
        prompts = {}
        prompts["prompt_sources"] = []
        prompts["prompt_targets"] = []

        for conv in examples:
            conv = conv["conversations"]

            # system
            if conv[0]["from"] != "system":
                prompt = self.system + self.default_system + self.end + '\n'
                start = 0
            elif conv[0]["from"] == "system" and conv[0]["value"] == "":
                prompt = self.system + self.default_system + self.end + '\n'
                start = 1
            else:
                prompt = self.system + conv[0]["value"] + self.end + '\n'
                start = 1

            for j in range(start, len(conv) - 1, 2):

                u = conv[j]["value"]
                ass = conv[j+1]["value"]
                prompt = prompt + self.user + u + self.end + '\n' + self.assistant
                response = ass + self.end
                prompts["prompt_sources"].append(prompt)
                prompts["prompt_targets"].append(response)

                prompt += response + '\n'

        return prompts

class SummarizationDataPreprocess:
    prompt_template = "\nSummarize the highlights of this article.\n"

    def tokenize_func(self, tokenizer, data_args, finetune_args):
        template_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.prompt_template))

        def preprocess_function(examples):

            articles = [q.strip() for q in examples["article"]]
            highlights = [q.strip() for q in examples["highlights"]]

            examples["input_ids"] = []
            examples["labels"] = []
            examples["attention_mask"] = []
            examples["decoder_input_ids"] = []
            examples["decoder_attention_mask"] = []
            examples["decoder_labels"] = []

            for article, highlight in zip(articles, highlights):
                max_input = data_args.max_source_length - len(template_ids)

                article_tokens = tokenizer.tokenize(article)[:max_input]
                prompt_ids = tokenizer.convert_tokens_to_ids(article_tokens) + template_ids

                # for inference
                decoder_input_ids = copy.deepcopy(prompt_ids)

                max_resp = data_args.max_seq_length - len(prompt_ids) - 1
                resp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(highlight))[:max_resp] + \
                        [tokenizer.eos_token_id]

                # for inference
                max_decoder_labels_len = data_args.max_seq_length - data_args.max_source_length - 1
                decoder_labels = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(highlight)
                        )[:max_decoder_labels_len] + [tokenizer.eos_token_id]

                input_ids = prompt_ids + resp_ids
                if not finetune_args.train_on_inputs:
                    labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids
                else:
                    labels = prompt_ids + resp_ids

                # padding
                input_len = len(input_ids)
                pad_len = data_args.max_seq_length - input_len
                input_ids = input_ids + [tokenizer.eos_token_id] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len
                attention_mask = [1] * input_len + [0] * pad_len

                assert len(input_ids) == data_args.max_seq_length
                assert len(prompt_ids) <= data_args.max_source_length
                assert len(labels) == len(input_ids) == len(attention_mask)

                examples["input_ids"].append(input_ids)
                examples["labels"].append(labels)
                examples["attention_mask"].append(attention_mask)

                # left padding for inference
                input_len = len(decoder_input_ids)
                pad_len = data_args.max_source_length - input_len
                decoder_input_ids = [tokenizer.eos_token_id] * pad_len + decoder_input_ids
                decoder_attention_mask = [0] * pad_len + [1] * input_len

                input_len = len(decoder_labels)
                pad_len = data_args.max_seq_length - data_args.max_source_length - input_len
                decoder_labels = decoder_labels + [IGNORE_INDEX] * pad_len
                examples["decoder_input_ids"].append(decoder_input_ids)
                examples["decoder_labels"].append(decoder_labels)
                examples["decoder_attention_mask"].append(decoder_attention_mask)


            return examples

        return preprocess_function


def preprocess_dataset(raw_datasets, tokenizer, data_args, finetune_args):

    if data_args.dataset_name == "Intel/orca_dpo_pairs":
        preprocess = IntelDpoDataPreprocess(
            data_args.dataset_name if data_args.dataset_name else data_args.train_file
        )
        for key in raw_datasets:
            prompts = preprocess.create_data(raw_datasets[key])
            columns_to_be_removed = list(raw_datasets[key].features.keys())
            raw_datasets[key] = raw_datasets[key].add_column(
                    "prompt_sources", prompts["source"]
                    )
            raw_datasets[key] = raw_datasets[key].add_column(
                    "prompt_targets", prompts["target"]
                    )
            raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)

        preprocess_fn = preprocess.tokenize_func(tokenizer, data_args, finetune_args)

    elif finetune_args.task == "chat":
        preprocess = ChatDataPreprocess(tokenizer.eos_token)
        new_datasets = datasets.DatasetDict()
        for key in raw_datasets:
            prompts = preprocess.create_data(raw_datasets[key])

            # deal irregular column name
            if "train" in key:
                new_key = "train"
            if "val" in key:
                new_key = "validation"
            if "test" in key:
                new_key = "test"

            new_datasets[new_key] = datasets.Dataset.from_dict(prompts)

        preprocess_fn = preprocess.tokenize_func(tokenizer, data_args, finetune_args)

        return new_datasets, preprocess_fn

    elif finetune_args.task == "SlimOrca":
        preprocess = SlimOrcaDataPreprocess(tokenizer.eos_token)
        new_datasets = datasets.DatasetDict()
        for key in raw_datasets:
            prompts = preprocess.create_data(raw_datasets[key])

            new_datasets[key] = datasets.Dataset.from_dict(prompts)

        preprocess_fn = preprocess.tokenize_func(tokenizer, data_args, finetune_args)

        return new_datasets, preprocess_fn

    elif finetune_args.task == "summarization":
        preprocess = SummarizationDataPreprocess()
        preprocess_fn = preprocess.tokenize_func(tokenizer, data_args, finetune_args)

    elif finetune_args.task == "completion" or finetune_args.task == "code-generation":
        # default use alpaca template
        preprocess = CompletionDataPreprocess(
            data_args.dataset_name if data_args.dataset_name else data_args.train_file
        )
        for key in raw_datasets:
            prompts = preprocess.create_data(raw_datasets[key])
            columns_to_be_removed = list(raw_datasets[key].features.keys())
            raw_datasets[key] = raw_datasets[key].add_column(
                    "prompt_sources", prompts["source"]
                    )
            raw_datasets[key] = raw_datasets[key].add_column(
                    "prompt_targets", prompts["target"]
                    )
            raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)

        preprocess_fn = preprocess.tokenize_func(tokenizer, data_args, finetune_args)

    else:
        raise NotImplementedError(f'finetune task data preprocessing is not support currently.')

    return raw_datasets, preprocess_fn
