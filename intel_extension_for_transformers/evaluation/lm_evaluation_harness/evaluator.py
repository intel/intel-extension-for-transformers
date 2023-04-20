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

import os
import random
import time
import numpy as np
import json

from lm_eval.base import LM, CachingLM
from lm_eval.tasks import get_task_dict
from lm_eval.utils import run_task_tests
from lm_eval.evaluator import evaluate as evaluate_func
from lm_eval.evaluator import make_table
from .models import huggingface
MODEL_REGISTRY = {
    "hf-causal": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,

}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]

def evaluate(model,
             model_args=None,
             tasks=[],
             new_fewshot=0,
             batch_size=None,
             device="cpu",
             no_cache=True,
             limit=None,
             bootstrap_iters=100000,
             description_dict=None,
             check_integrity=False,
             decontamination_ngrams_path=None,
             seed=1234,
             user_model=None,
            ):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.
        EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param seed: Optional
        Set seed
    :param user_model: Optional[Object]
        Model object user provided.
    :param output_dir: str
        Save the results Path
    :return
        Dictionary of results
    """
    random.seed(seed)
    np.random.seed(seed)

    assert tasks != [], "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = get_model(model).create_from_arg_string(
            model_args, {"batch_size": batch_size, "device": device}
        )
    else:
        assert isinstance(model, LM)
        lm = model

    if not no_cache:
        lm = CachingLM(
            lm,
            "lm_cache/"
            + model
            + "_"
            + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
            + ".db",
        )
    
    task_dict = get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)
    
    if user_model:
        lm.model = user_model

    results = evaluate_func(
        lm=lm,
        task_dict=task_dict,
        provide_description=None,
        num_fewshot=new_fewshot,
        bootstrap_iters=bootstrap_iters,
        limit=limit,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path
    )

    print(make_table(results)) 
    return results
