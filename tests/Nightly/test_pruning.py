# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import shutil
import torch.utils.data as data
import unittest
from intel_extension_for_transformers.transformers import (
    metrics,
    OptimizedModel,
)
from neural_compressor.config import WeightPruningConfig
from intel_extension_for_transformers.transformers.trainer import NLPTrainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

os.environ["WANDB_DISABLED"] = "true"



class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b)
        self.encoded_dict['labels'] = 1

    def __len__(self):
        return 1

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.encoded_dict


class TestPruning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased'
        )
        self.dummy_dataset = DummyDataset()
        self.trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./pruned_model', ignore_errors=True)

    def test_fx_model_prune(self):
        origin_weight = copy.deepcopy(self.model.classifier.weight)

        self.trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )
        metric = metrics.Metric(name="eval_loss")
        self.trainer.metrics = metric
        pruning_conf = WeightPruningConfig([{"start_step": 0, "end_step": 2}],
                                            target_sparsity=0.64,
                                            pruning_scope="local")
        pruned_model = self.trainer.prune(pruning_config=pruning_conf)
        # By default, model will be saved in tmp_trainer dir.
        self.trainer.save_model('./pruned_model')
        loaded_model = OptimizedModel.from_pretrained(
            './pruned_model',
        )
        pruned_weight = copy.deepcopy(pruned_model.classifier.weight)
        loaded_weight = copy.deepcopy(loaded_model.classifier.weight)
        # check pruned model
        self.assertTrue((pruned_weight != origin_weight).any())
        # check loaded model
        self.assertTrue((pruned_weight == loaded_weight).all())

    def test_functional_prune(self):
        def eval_func(model):
            return 1

        def train_func(model):
            return model

        self.trainer = NLPTrainer(self.model)
        pruning_conf = WeightPruningConfig([{"start_step": 0, "end_step": 2}],
                                            target_sparsity=0.64,
                                            pruning_scope="local")
        self.trainer.prune(pruning_conf,
                           provider="inc",
                           train_func = train_func,
                           eval_func = eval_func,)


if __name__ == "__main__":
    unittest.main()
