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

from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from optimum.habana import GaudiConfig, GaudiTrainer
from intel_extension_for_transformers.transformers.dpo_trainer import DPOTrainer
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

def is_peft_available():
    import importlib
    return importlib.util.find_spec("peft") is not None

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): # pragma: no cover
            module.p = 0

class GaudiDPOTrainer(DPOTrainer, GaudiTrainer):
    r"""
    Initialize DPOTrainer, refer: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head.
            Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will
            create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified,
            the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch,
            given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
            The callbacks to use for training.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch.
            This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration,
            the model will be wrapped in a PEFT model.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        beta: float = 0.1,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        disable_dropout: bool = True,
    ):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        self.ref_model = ref_model

        if disable_dropout: # pragma: no cover
            disable_dropout_in_model(model)
            disable_dropout_in_model(self.ref_model)

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        args.remove_unused_columns = False
        gaudi_config = GaudiConfig()
        gaudi_config.use_fused_adam = True
        gaudi_config.use_fused_clip_norm = True

        GaudiTrainer.__init__(
            self,
            model=model,
            gaudi_config=gaudi_config,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        if self.is_deepspeed_enabled: # pragma: no cover
            # Read more about the issue in https://github.com/huggingface/trl/pull/687
            self.ref_model = self.accelerator._prepare_deepspeed(self.ref_model)[0]
            self.ref_model.eval()
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
