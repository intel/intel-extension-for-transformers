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


from transformers import (
    AutoModelForSequenceClassification,
    Trainer
)

from infer import DlsaInference
from utils import compute_metrics, save_test_metrics


class TrainerInfer(DlsaInference):
    def _load_data(self):
        return super()._load_data()

    def _preprocess(self):
        return super()._preprocess()

    def _load_model(self):
        with self.track('Load Model'):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name_or_path)

            self.trainer = Trainer(
                model=self.model,  # the instantiated HF model to be trained
                args=self.training_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # evaluation metrics
                tokenizer=self.tokenizer
            )

    def _do_infer(self):
        test_metrics = ""
        if self.training_args.do_predict:
            with self.track('Inference'):
                preds, _, metrics = self.trainer.predict(self.test_data)
                test_metrics = save_test_metrics(metrics, len(self.test_data), self.training_args.output_dir)
        print(test_metrics)
