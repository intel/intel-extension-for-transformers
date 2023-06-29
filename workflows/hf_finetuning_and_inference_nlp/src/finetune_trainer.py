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
from os import path
from finetune import DlsaFinetune
from utils import compute_metrics, save_performance_metrics, save_train_metrics



class FinetuneTrainer(DlsaFinetune):
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
                train_dataset=self.train_data,  # training dataset
                compute_metrics=compute_metrics,  # evaluation metrics
                tokenizer=self.tokenizer
            )

    def _do_finetune(self):
        if self.training_args.do_train:
            with self.track('Fine-Tune'):
                train_result = self.trainer.train()
                self.trainer.save_model()
                save_train_metrics(train_result, self.trainer, len(self.train_data))

    def _do_infer(self):
        if self.training_args.do_predict:
            with self.track('Inference'):
                if not self.args.save_detailed_performance_metrics:
                    preds, _, metrics = self.trainer.predict(self.test_data)
                    print(
                            f"\n*********** TEST_METRICS ***********\nAccuracy: {metrics['test_acc']}\n"
                        )
                else:
                    save_performance_metrics(self.trainer, self.train_data, 
                                path.join(self.training_args.output_dir, self.args.finetune_output) )
