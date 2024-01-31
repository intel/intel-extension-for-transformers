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

import os
import shutil
import unittest

import torch.utils.data as data
from transformers import AutoModelForPreTraining, AutoTokenizer

from intel_extension_for_transformers.transformers import NASConfig, metrics
from intel_extension_for_transformers.transformers.trainer import NLPTrainer

os.environ["WANDB_DISABLED"] = "true"


class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b)
        self.encoded_dict["labels"] = [-100] * len(self.encoded_dict["input_ids"])
        self.encoded_dict["labels"][1] = 17953
        self.encoded_dict["next_sentence_label"] = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.encoded_dict


class TestNAS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForPreTraining.from_pretrained("prajjwal1/bert-tiny")
        self.dummy_dataset = DummyDataset()
        self.trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./tmp_trainer", ignore_errors=True)

    def test_nas(self):
        for search_algorithm in ["BO", "Grid", "Random"]:
            max_trials = 6 if search_algorithm == "Random" else 3
            nas_config = NASConfig(
                search_space={
                    "hidden_size": [128, 256],
                    "intermediate_size": [256, 512],
                },
                search_algorithm=search_algorithm,
                max_trials=max_trials,
                metrics=metrics.Metric(name="eval_loss", greater_is_better=False),
                seed=42,
            )
            framework = nas_config.framework
            search_space = nas_config.search_space
            search_algorithm = nas_config.search_algorithm
            max_trials = nas_config.max_trials
            metric = nas_config.metrics
            seed = nas_config.seed
            best_model_archs = self.trainer.nas(
                nas_config, model_cls=AutoModelForPreTraining
            )
            # check best model architectures
            self.assertTrue(len(best_model_archs) > 0)


if __name__ == "__main__":
    unittest.main()
