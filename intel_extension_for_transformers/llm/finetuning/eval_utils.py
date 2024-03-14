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

import evaluate
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Union
import time
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
import math
from transformers import TrainerCallback

@torch.no_grad()
def compute_rouge_metric(model, tokenizer, eval_dataset, training_args, gen_kwargs):
    model.eval()
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Metric
    metric = evaluate.load("rouge")

    def collate_fn(batch):
        input_ids = [torch.tensor(ins["decoder_input_ids"]) for ins in batch]
        labels = [torch.tensor(ins["decoder_labels"]) for ins in batch]
        attention_mask = [torch.tensor(ins["decoder_attention_mask"]) for ins in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                )

    # TODO: support batch_size >1
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn,
                            batch_size=1)


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    for step, batch in enumerate(eval_dataloader):
        preds = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                **gen_kwargs,
                )
        labels = batch["labels"]
        labels = labels.cpu().numpy()

        preds = preds.cpu().numpy()

        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id).tolist()
        # only pred
        preds = [pred[batch["input_ids"].shape[1]:] for pred in preds]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).tolist()
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
                )


    result = metric.compute(use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result

def evaluate_plus_ppl(
    self,
    eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
) -> Dict[str, float]:
    """
    Copied from Trainer.evaluate:
    https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/trainer.py#L3029
    The only differences are:
    - add new metric eval_ppl
    """
    # handle multiple eval datasets
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    if isinstance(eval_dataset, dict):
        metrics = {}
        for eval_dataset_name, _eval_dataset in eval_dataset.items():
            dataset_metrics = self.evaluate(
                eval_dataset=_eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
            )
            metrics.update(dataset_metrics)
        return metrics

    # memory metrics - must set up as early as possible
    self._memory_tracker.start()

    eval_dataloader = self.get_eval_dataloader(eval_dataset)

    start_time = time.time()

    eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
    output = eval_loop(
        eval_dataloader,
        description="Evaluation",
        # No point gathering the predictions if there are no metrics, otherwise we defer to
        # self.args.prediction_loss_only
        prediction_loss_only=True if self.compute_metrics is None else None,
        ignore_keys=ignore_keys,
        metric_key_prefix=metric_key_prefix,
    )

    total_batch_size = self.args.eval_batch_size * self.args.world_size
    if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
    output.metrics.update(
        speed_metrics(
            metric_key_prefix,
            start_time,
            num_samples=output.num_samples,
            num_steps=math.ceil(output.num_samples / total_batch_size),
        )
    )

    output.metrics[f"{metric_key_prefix}_ppl"] = math.exp(output.metrics[f"{metric_key_prefix}_loss"])

    self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

    self.log(output.metrics)

    self._memory_tracker.stop_and_update_metrics(output.metrics)

    return output.metrics


class LMEvalCallback(TrainerCallback):
    def __init__(self, lm_eval_func, device=None):
        self.lm_eval = lm_eval_func
        self.device = device
        self.warmup = True

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return
        if self.device == "hpu":
            results = self.lm_eval(user_model=kwargs["model"],
                    user_tokenizer=kwargs["tokenizer"],
                    warmup=self.warmup)
            self.warmup = False
        else:
            results = self.lm_eval(model="simple-hf-causal",
                    user_model=kwargs["model"],
                    user_tokenizer=kwargs["tokenizer"],
                    warmup=False)
        task_metrics = {}
        for task_name in results["results"]:
            for metric in results["results"][task_name]:
                if "stderr" in metric:
                    continue
                metric_name = task_name + "_" + metric
                task_metrics[metric_name] = results["results"][task_name][metric]
        kwargs["metrics"].update(task_metrics)
