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

import os
import sys
from transformers import TrainingArguments, HfArgumentParser
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    BaseFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model
from intel_extension_for_transformers.neural_chat.utils.common import is_hpu_available

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    if not is_hpu_available:
        parser = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, FinetuningArguments)
        )
    else:
        from optimum.habana import GaudiTrainingArguments

        parser = HfArgumentParser(
            (ModelArguments, DataArguments, GaudiTrainingArguments, FinetuningArguments)
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

    finetune_cfg = BaseFinetuningConfig(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetune_args=finetune_args,
    )
    finetune_model(finetune_cfg)

if __name__ == "__main__":
    main()
