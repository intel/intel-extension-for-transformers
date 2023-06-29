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
from torch.utils.data import DataLoader
from os import path
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)
from intel_extension_for_transformers.optimization import (
    QuantizationConfig,
    PruningConfig,
    PrunerConfig,
    metrics,
    objectives,
)
from intel_extension_for_transformers.optimization.trainer import (
    NLPTrainer,
)
from finetune import DlsaFinetune
from utils import PredsLabels, compute_metrics, save_train_metrics, save_test_metrics, save_performance_metrics


class FinetuneItrex(DlsaFinetune):
    def _load_data(self):
        return super()._load_data()

    def _preprocess(self):
        return super()._preprocess()

    def _load_model(self):
        with self.track("Load Model"):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path, num_labels=self.num_labels if self.num_labels is not None else None
            )

    def _do_finetune(self):
        if self.training_args.do_train:
            with self.track("Fine-Tune"):
                self.training_args.use_ipex = vars(self.args).get("use_ipex", False)
                if self.args.dtype_ft == "bf16":
                    self.training_args.bf16 = True
                self.trainer = NLPTrainer(
                    model=self.model,  # the instantiated HF model to be trained
                    args=self.training_args,  # training arguments, defined above
                    train_dataset=self.train_data,  # training dataset
                    eval_dataset=self.test_data,
                    compute_metrics=compute_metrics,  # evaluation metrics
                    tokenizer=self.tokenizer,
                )

                if self.args.dtype_ft == "bf16" and not (self.training_args.use_ipex or vars(self.args).get("use_onednn", True)):
                    raise ValueError("BF16 with both IPEX and OneDNN disabled is currently not implemented...")

                with torch.backends.mkldnn.flags(enabled = self.training_args.use_ipex or vars(self.args).get("use_onednn", True)):
                    train_result = self.trainer.train()
                
                self.trainer.save_model()
                
                save_train_metrics(train_result, self.trainer, len(self.train_data))
               
    def _do_infer(self):
        with torch.backends.mkldnn.flags(enabled = self.training_args.use_ipex or vars(self.args).get("use_onednn", True)):
            if self.training_args.do_predict:
                with self.track("Inference"):
                    if not self.args.save_detailed_performance_metrics:
                        preds, _, metrics = self.trainer.predict(self.test_data)
                        print(
                            f"\n*********** TEST_METRICS ***********\nAccuracy: {metrics['test_acc']}\n"
                        )
                    else:
                        save_performance_metrics(self.trainer, self.train_data,
                                path.join(self.training_args.output_dir, self.args.finetune_output) )
