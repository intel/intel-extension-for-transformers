# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

import numpy as np
import torch
from os import path
from intel_extension_for_transformers.optimization import (
    QuantizationConfig,
    metrics,
    objectives,
)
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)

from infer import DlsaInference
from utils import PredsLabels, compute_metrics, save_performance_metrics


class ItrexInfer(DlsaInference):
    def _load_data(self):
        return super()._load_data()

    def _preprocess(self):
        return super()._preprocess()

    def _load_model(self):
        with self.track("Load Model"):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path
            )

        if self.args.dtype_inf == "fp32":
            self.training_args.use_ipex = vars(self.args).get("use_ipex", False)
            self.trainer = NLPTrainer(
                model=self.model,  # the instantiated HF model to be trained
                args=self.training_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # evaluation metrics
                tokenizer=self.tokenizer,
            )

        elif self.args.dtype_inf == "bf16":
            self.training_args.use_ipex = vars(self.args).get("use_ipex", False)
            self.training_args.bf16 = True
            self.trainer = NLPTrainer(
                model=self.model,  # the instantiated HF model to be trained
                args=self.training_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # evaluation metrics
                tokenizer=self.tokenizer,
            )

        elif self.args.dtype_inf == "int8":
            self.trainer = NLPTrainer(
                model=self.model,  # the instantiated HF model to be trained
                args=self.training_args,  # training arguments, defined above
                eval_dataset=self.test_data,  # using the test dataset for evaluation
                compute_metrics=compute_metrics,  # evaluation metrics
                tokenizer=self.tokenizer,
            )

            metric = metrics.Metric(name="eval_acc", is_relative=True, criterion=0.03)
            q_config = QuantizationConfig(
                framework="pytorch",
                approach="PostTrainingStatic",
                max_trials=200,  # set the Max tune times
                metrics=[metric],
                objectives=[objectives.performance],
            )
            eval_dataloader = self.trainer.get_eval_dataloader()
            self.model = self.trainer.quantize(
                quant_config=q_config, calib_dataloader=eval_dataloader
            )
            
        else:
            error_msg = f"Now only support fp32, bf16 and int8.Your input datatype is {self.args.dtype_inf}."
            raise ValueError(error_msg)

    def _do_infer(self):
        
        if self.args.dtype_inf == "bf16" and not (self.training_args.use_ipex or vars(self.args).get("use_onednn", True)):
                    raise ValueError("BF16 with both IPEX and OneDNN disabled is currently not implemented...")

        with torch.backends.mkldnn.flags(enabled = self.training_args.use_ipex or vars(self.args).get("use_onednn", True)):

            if self.training_args.do_predict:
                with self.track("Inference"):
                    if not self.args.save_detailed_performance_metrics:
                        preds, _, metrics = self.trainer.predict(self.test_data)
                        print(
                            f"\n*********** TEST_METRICS ***********\nAccuracy: {metrics['test_acc']}\n"
                        )
                    else:
                        save_performance_metrics(self.trainer, self.test_data,
                                            path.join(self.training_args.output_dir, self.args.inference_output) )
