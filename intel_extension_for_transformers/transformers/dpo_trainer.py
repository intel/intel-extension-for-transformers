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

import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import importlib
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
import time
import logging
from intel_extension_for_transformers.utils.device_utils import is_hpu_available

logger = logging.getLogger(__name__)

def is_peft_available():
    return importlib.util.find_spec("peft") is not None

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): # pragma: no cover
            module.p = 0

class DPOTrainer(Trainer):
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
        super().__init__(
            model=model,
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

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities.
               Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token.
               Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log
               probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape: # pragma: no cover
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1)

    def dpo_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        len_chosen = batch["input_ids"].shape[0] // 2

        model_kwargs = {}

        all_logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            batch["labels"],
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.dpo_forward(model, batch)

        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.dpo_forward(self.ref_model, batch)
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else "" # pragma: no cover
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process: # pragma: no cover
            self.store_metrics(metrics, train_eval="train")

        if return_outputs: # pragma: no cover
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval" # pragma: no cover
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        # pylint: disable=E1101
        return super().log(logs)


if is_hpu_available: # pragma: no cover
    # pylint: disable=E0611
    from optimum.habana import GaudiConfig, GaudiTrainer # pylint: disable=E0401
    class GaudiDPOTrainer(DPOTrainer, GaudiTrainer):
        r"""Initialize habana
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
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            self.ref_model.eval()

            if args.use_hpu_graphs_for_training:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph # pylint: disable=E0611, E0401
                ref_model = self.accelerator.unwrap_model(self.ref_model)
                ref_model = wrap_in_hpu_graph(ref_model)
