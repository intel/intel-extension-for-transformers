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
"""Data augmentation API.

Please refer to https://github.com/intel/intel-extension-for-transformers/blob/main/docs/data_augmentation.md.
"""

import csv
import json
import math
import os
from enum import Enum
from operator import methodcaller

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from intel_extension_for_transformers.transformers.utils.utility import LazyImport

torch = LazyImport("torch")

DEFAULT_OUTPUT_FILE = "augmented_dataset"

EOS = "</s>"


class AugmenterType(Enum):
    """Enumeration of types of augmentation."""

    TEXTGENERATIONAUG = "textgenerationaug"
    KEYBOARDAUG = "KeyboardAug"
    OCRAUG = "OcrAug"
    SPELLINGAUG = "SpellingAug"
    CONTEXTUALWORDEMBSFORSENTENCEAUG = "ContextualWordEmbsForSentenceAug"


AUGMENTER_MAPPING = {
    AugmenterType.KEYBOARDAUG.value: nac,
    AugmenterType.OCRAUG.value: nac,
    AugmenterType.SPELLINGAUG.value: naw,
    AugmenterType.CONTEXTUALWORDEMBSFORSENTENCEAUG.value: nas,
}


def get_augmenter_from_type(aug_type: str):
    """Get nlpaug's augmenter by augment_type name.

    The nlpaug is a library helps you with augmenting nlp for your machine learning projects.
    It provide many augmenter, please refer to https://github.com/makcedward/nlpaug#augmenter.
    """
    assert aug_type in AUGMENTER_MAPPING, "Unsupported the augmenter type:{}".format(
        aug_type
    )
    return AUGMENTER_MAPPING[aug_type]


class DataAugmentation:
    """DataAugmentation provides many ways to enhance existing datasets.

    Args:
            augmenter_type (string): augmenter type like: "TextGenerationAug", "KeyboardAug", ...

    Example::

        aug = DataAugmentation(augmenter_type="TextGenerationAug")
        aug.input_dataset = self.origin_data
        aug.output_path = os.path.join(self.result_path, "test1.cvs")
        aug.augmenter_arguments = {'model_name_or_path': 'gpt2-medium'}
        aug.data_augment()
    """

    def __init__(self, augmenter_type: str):
        """Init an instance of DataAugmentation."""
        self._augmenter_type = AugmenterType[augmenter_type.upper()].value
        self._output_path = "save_path/augmented_dataset.csv"
        self._input_dataset = None
        self._data_config_or_task_name = None
        self._augmenter_arguments = None
        self._column_names = "sentence"
        self._split = "validation"
        self._num_samples = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def augmenter_type(self):
        """Get augmenter type."""
        return self._augmenter_type

    @augmenter_type.setter
    def augmenter_type(self, augmenter_type):
        # pylint: disable=C0301
        """Set augmenter type.

        Please refer to https://github.com/intel/intel-extension-for-transformers/blob/main/docs/data_augmentation.md#supported-augmenter
        """
        self._augmenter_type = AugmenterType[augmenter_type.upper()].value

    @property
    def output_path(self):
        """Get output path for new datasets."""
        return self._output_path

    @output_path.setter
    def output_path(self, output_path):
        """Set output path for new datasets."""
        self._output_path = output_path

    @property
    def input_dataset(self):
        """Get path for original datasets."""
        return self._input_dataset

    @input_dataset.setter
    def input_dataset(self, input_dataset):
        # pylint: disable=C0301
        """Set original datasets.

        Input_dataset can be a file path, the file should be contained the samples as format: 'label' + '\t' + 'sentence' + EOS + '\n',please refer to
        https://github.com/intel/intel-extension-for-transformers/blob/1d20bcafc3a1f5953400c0c42dafbaa3f700ac1d/tests/test_data_augmentation.py#L9.
        Input_dataset also can be a huggingface datasets name, like: "glue"...,
        if the datasets has task name or data config, you should set data_config_or_task_name:
            example::

                aug.input_dataset = "glue"
                aug.data_config_or_task_name = "sst2"
        """
        self._input_dataset = input_dataset

    @property
    def data_config_or_task_name(self):
        """Get task name for glue datasets or datasets configuration."""
        return self._data_config_or_task_name

    @data_config_or_task_name.setter
    def data_config_or_task_name(self, data_config_or_task_name):
        """Set task name for glue datasets or datasets configuration."""
        self._data_config_or_task_name = data_config_or_task_name

    @property
    def column_names(self):
        """Get column name of datasets."""
        return self._column_names

    @column_names.setter
    def column_names(self, column_names: str):
        """Set column name of datasets."""
        self._column_names = column_names

    @property
    def num_samples(self):
        """Get number of samples generated by augmenter for every original sample."""
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples: int):
        """Set number of samples generated by augmenter for every original sample."""
        self._num_samples = num_samples

    @property
    def augmenter_arguments(self):
        """Get the parameters of augmenter."""
        return self._augmenter_arguments

    @augmenter_arguments.setter
    def augmenter_arguments(self, augmenter_arguments):
        # pylint: disable=C0301
        """Set parameters dict for augmenters. Different augmenter has different parameters.

        For TextGenerationAug type, it is huggingface model name or a path, example: {'model_name_or_path': 'gpt2-medium'}.
        For KeyboardAug type, it is parameters which defined in ["KeyboardAug"](https://github.com/makcedward/nlpaug/blob/40794970124c26ce2e587e567738247bf20ebcad/nlpaug/augmenter/char/keyboard.py#L46).
            example::

                {"aug_char_min": 1, "aug_char_max": 10, "aug_char_p": 0.3, "aug_word_p": 0.3, "aug_word_min": 1, "aug_word_max": 10,
                 "stopwords": None, "tokenizer": None, "reverse_tokenizer": None, "include_special_char": True, "include_numeric": True,
                 "include_upper_case": True, "lang": "en", "verbose": 0, "stopwords_regex": None, "model_path": None, "min_char": 4}
        Please refer [data_augmentation document](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/data_augmentation.md#supported-augmenter)
        """
        self._augmenter_arguments = augmenter_arguments

    @property
    def custom_augmenter(self):
        """Get augmenter which be set by customer."""
        return self._custom_augmenter

    @custom_augmenter.setter
    def custom_augmenter(self, custom_augmenter):
        """Set customer data augmenter."""
        self._custom_augmenter = custom_augmenter

    def data_augment(self):
        """Execute the process of data augmentation."""
        assert self._input_dataset is not None, (
            "Please pass the dataset name or "
            "A csv or a json file to DataAugmentation.input_dataset."
        )

        assert self._column_names is not None, (
            "Please pass column names "
            "which you want to augmentation to DataAugmentation.column_names"
        )

        extension = None
        if os.path.isfile(self._input_dataset):
            extension = self._input_dataset.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`input dataset` should be a csv or a json file."
            if self._input_dataset.endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset(
                    "csv", data_files=self._input_dataset, delimiter="\t", split="train"
                )
            else:  # pragma: no cover
                # Loading a dataset from local json files
                raw_datasets = load_dataset("json", data_files=self._input_dataset)
        else:  # pragma: no cover
            if self._input_dataset == "glue":
                assert (
                    self._data_config_or_task_name is not None
                ), "Please pass the task name to DataAugmentation.data_config_or_task_name."
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self._input_dataset, self._data_config_or_task_name, split=self._split
            )
        if extension is None:  # pragma: no cover
            extension = "csv"

        if os.path.isfile(self._output_path):
            self._output_path = self._output_path
        elif os.path.isdir(self._output_path):
            self._output_path = os.path.join(
                self._output_path, DEFAULT_OUTPUT_FILE + "." + extension
            )
        else:
            path, name = os.path.split(self._output_path)
            os.makedirs(path, exist_ok=True)
            self._output_path = os.path.join(
                path,
                name if name is not None else DEFAULT_OUTPUT_FILE + "." + extension,
            )

        if self._augmenter_type == AugmenterType.TEXTGENERATIONAUG.value:
            self.text_generation_augmentation(extension, raw_datasets)
        else:
            self.mit_data_augmentation(extension, raw_datasets)

    def text_generation_augmentation(self, extension, raw_datasets):
        """Execute the process of text generation augmentation.

        Args:
            extension: No used
            raw_datasets: The original datasets, the datasets can be from huggingface datasets(like: glue/sst2) or
            the customer datasets, each sample should be:
                'label' + '\t' + 'sentence' + EOS + '\n'

        augmenter_arguments for text generation augmentation::

            {'model_name_or_path': 'gpt2'," \
             'k': 0, // top_k, default: 0
             'p': 0.9, // top_p, default: 0.9
             'temperature': 1.0, // temperature of 1.0 has no effect,
                                 // lower tend toward greedy sampling, default: 1.0
             'repetition_penalty': 1.0, // primarily useful for CTRL model, default: 1.0
             'num_return_sentences': -1 // number of sentences to generate. default is -1 means the entire dataset
             'num_samples': 1, // number of samples generated by each conditional text
                               // generation run. default: 1
             'stop_token': EOS // end of sentence token, at which text generation is stopped.
                              // default: EOS
            }
        """
        assert self._augmenter_arguments is not None, (
            "Please pass a pretrained model name or path to "
            "DataAugmentation.augmenter_arguments like: "
            "{'model_name_or_path': 'gpt2',"
            "......}"
        )
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            CTRLLMHeadModel,
            CTRLTokenizer,
            GPT2LMHeadModel,
            GPT2Tokenizer,
            OpenAIGPTLMHeadModel,
            OpenAIGPTTokenizer,
            TransfoXLLMHeadModel,
            TransfoXLTokenizer,
            XLMTokenizer,
            XLMWithLMHeadModel,
            XLNetLMHeadModel,
            XLNetTokenizer,
            pipeline,
        )

        MODEL_CLASSES = {
            "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
            "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
            "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
            "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
            "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
            "xlm": (XLMWithLMHeadModel, XLMTokenizer),
        }

        model_name_or_path = self._augmenter_arguments.get("model_name_or_path", None)
        stop_token = self._augmenter_arguments.get("stop_token", EOS)
        num_return_sentences = self._augmenter_arguments.get("num_return_sentences", -1)

        assert model_name_or_path is not None, (
            "Please pass a pretrained model name or path to "
            "DataAugmentation.augmenter_arguments like: "
            "{'model_name_or_path': 'gpt2',"
            "......}"
        )
        config = AutoConfig.from_pretrained(model_name_or_path)
        assert (
            config.model_type in MODEL_CLASSES
        ), "Unsupported this model to augment data:{}".format(config.model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )
        model.to(self.device)

        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1 if self.device == torch.device("cpu") else 0,
        )
        assert os.path.isfile(
            self._input_dataset
        ), "Please ensure the input dataset is a txt file"
        # map between the prefix and its number of occurrences in the input file
        label2count = dict()
        label2count[0] = 0
        label2count[1] = 0

        lengths = list()

        for sentence in raw_datasets:
            label2count[sentence["label"]] += 1
            length = len(sentence["sentence"].split())
            lengths.append(length)

        from statistics import mean

        m = int(mean(lengths))
        std = int(np.std(lengths)) + 1
        max_length = m + std
        min_length = m - std

        total_count = sum(label2count.values())
        factor = (
            total_count
            if num_return_sentences <= 0
            else int(math.ceil(num_return_sentences / self._num_samples))
        )
        p0 = label2count[0] / total_count
        p1 = 1 - p0

        prefix_texts = np.random.choice([0, 1], size=(factor,), p=[p0, p1])
        augmented = set()

        with open(self._output_path, "w", encoding="utf8") as file:
            file.write("label" + "\t" + "sentence" + "\n")
            for prefix in tqdm(prefix_texts):
                loops = 0
                while loops < 2:
                    text_inputs = str(prefix) + "\t"
                    output_sequences = text_generator(
                        text_inputs=text_inputs,
                        early_stopping=True,
                        temperature=self._augmenter_arguments.get("temperature", 1.0),
                        top_k=self._augmenter_arguments.get("k", 0),
                        top_p=self._augmenter_arguments.get("p", 0.9),
                        repetition_penalty=self._augmenter_arguments.get(
                            "repetition_penalty", 1.0
                        ),
                        do_sample=True,
                        num_return_sequences=self._num_samples,
                        clean_up_tokenization_spaces=True,
                        return_full_text=True,
                        max_length=max_length,
                        min_length=min_length,
                    )
                    l_text_inputs = len(text_inputs.strip())
                    for seq in output_sequences:
                        text = seq["generated_text"]
                        text = text[
                            : text.find(stop_token)
                            if stop_token and text.find(stop_token) > -1
                            else None
                        ].strip()
                        text = text[
                            : text.find("\n") if text.find("\n") > -1 else None
                        ].strip()
                        if len(text) > l_text_inputs and text not in augmented:
                            file.write(text + "\n")
                            augmented.add(text)
                            loops += 1
                    loops += 1

                    file.flush()

    def mit_data_augmentation(self, extension, raw_datasets):  # pragma: no cover
        """Execute the process of nlpaug data augmentation.

        nlpaug is the library helps you with augmenting nlp for your machine learning projects.
        It provide many augmenter, please refer to: https://github.com/makcedward/nlpaug#augmenter.

        Args:
            extension (str): Suffix name of the original dataset file, now only support "csv" and "json".
            raw_datasets (dict): original datasets, like:label + '\t' + sentence.
                user should set column_names if the sample contains multiple sentences:
                    aug.column_names = ['sentence1', 'sentence2'].
        """
        column_names = raw_datasets.column_names
        if "idx" in column_names:
            column_names.remove("idx")
        args = {} if self._augmenter_arguments is None else self._augmenter_arguments

        aug = get_augmenter_from_type(self._augmenter_type)
        auger = methodcaller(self._augmenter_type, **args)(aug)

        with open(self._output_path, "w", encoding="utf8") as file:
            if extension == "json":
                writer = json
            else:
                writer = csv.DictWriter(file, fieldnames=column_names, delimiter="\t")
                writer.writeheader()
            for sample in raw_datasets:
                if "idx" in sample:
                    sample.pop("idx")
                if isinstance(self._column_names, list):
                    for column in self._column_names:
                        text = auger.augment(sample[column], n=self._num_samples)
                        sample[column] = text
                else:
                    text = auger.augment(
                        sample[self._column_names], n=self._num_samples
                    )
                    sample[self._column_names] = text
                if extension == "json":
                    text = writer.dumps(sample)
                    file.write(text)
                    file.write("\n")
                else:
                    writer.writerow(sample)
                file.flush()
