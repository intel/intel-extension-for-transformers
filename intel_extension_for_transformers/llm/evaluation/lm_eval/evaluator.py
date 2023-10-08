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

import json
import logging
import os

from lm_eval import evaluator, utils
from intel_extension_for_transformers.transformers.utils import logger

class LMParameters:
    model_args = ""
    tasks = "lambada_openai"
    provide_description = False
    num_fewshot = 0
    batch_size = 1
    max_batch_size = None
    device = None
    output_path = None
    limit = None
    data_samping = None
    no_cache = False
    decontamination_ngrams_path = None
    description_dict_path = None
    check_integrity = False
    write_out = False
    output_base_path = None


args = LMParameters()
def evaluate(model,
             tasks=None,
             batch_size=None,
             device=None,
             limit=None,
             no_cache=True):

    if batch_size is not None:
        args.batch_size = batch_size
    if device is None:
        args.device = "cpu"
    else:
        args.device = device
    if limit is not None:
        args.limit = limit
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    if no_cache:
        args.no_cache = no_cache
        logger.info("no_cache is used for lm_eval evaluation.")

    results = evaluator.simple_evaluate(
        model=model,
        model_args=args.model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        dirname = os.path.dirname(args.output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(evaluator.make_table(results))

    return results
