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
import sys
import unittest

sys.path.insert(0, './')

import torch
import numpy as np
from intel_extension_for_transformers.transformers.pruner import WeightPruningConfig, Pruning
from intel_extension_for_transformers.transformers.utils import SparsityConfig
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def build_fake_yaml_basic():
    fake_snip_yaml = """
version: 1.0

model:
  name: "bert-mini"
  framework: "pytorch"

pruning:
  approach:
    weight_compression_v2:
      start_step: 0
      end_step: 0

      pruning_op_types: ["Linear"]
      target_sparsity: 0.5
      max_sparsity_ratio_per_op: 0.5

      pruners:
        - !PrunerV2
            excluded_op_names: ["classifier", "pooler", ".*embeddings*"]
            pattern: "2:4"
            pruning_frequency: 50
            pruning_scope: "global"
            pruning_type: "snip_momentum"
            sparsity_decay_type: "linear"
    """
    with open('fake_snip.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_snip_yaml)

class TestPruning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = torchvision.models.resnet18()
        build_fake_yaml_basic()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_snip.yaml')

    def test_pruning_basic(self):
        local_configs = [
             {
                "op_names": ['layer1.*'],
                'target_sparsity': 0.5,
                "pattern": '8x2',
                "pruning_type": "magnitude_progressive",
                "false_key": "this is to test unsupported keys"
            },
            {
                "op_names": ['layer2.*'],
                'target_sparsity': 0.5,
                'pattern': '2:4'
            },
            {
                "op_names": ['layer3.*'],
                'target_sparsity': 0.7,
                'pattern': '5x1',
                "pruning_type": "snip_progressive"
            }
        ]
        config = WeightPruningConfig(
            local_configs,
            target_sparsity=0.8,
            start_step=1,
            end_step=10
        )
        prune = Pruning(config)
        prune.model = self.model

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        x_train = np.random.uniform(low=0., high=1., size=tuple([10, 3, 224, 224]))
        y_train = np.random.randint(low=0, high=2, size=tuple([10]))
        x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
                                torch.tensor(y_train, dtype=torch.long)
        dummy_dataset = TensorDataset(x_train, y_train)
        dummy_dataloader = DataLoader(dummy_dataset)

        prune.on_train_begin()
        prune.update_config(pruning_frequency=4)
        for epoch in range(2):
            self.model.train()
            prune.on_epoch_begin(epoch)
            local_step = 0
            for image, target in dummy_dataloader:
                prune.on_step_begin(local_step)
                output = self.model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                prune.on_before_optimizer_step()
                optimizer.step()
                prune.on_after_optimizer_step()
                prune.on_step_end()
                local_step += 1

            prune.on_epoch_end()
        prune.get_sparsity_ratio()
        prune.on_train_end()
        prune.on_before_eval()
        prune.on_after_eval()

    def test_pruning_pattern(self):
        self.model = torchvision.models.resnet18()
        local_configs = [
            {
                "op_names": ['layer1.*'],
                'target_sparsity': 0.5,
                "pattern": '5:8',
                "pruning_type": "magnitude"
            },
            {
                "op_names": ['layer2.*'],
                "pattern": '1xchannel',
                "pruning_scope": "global"
            },
            {
                "start_step": 2,
                "end_step": 20,
                "op_names": ['layer3.*'],
                'target_sparsity': 0.666666,
                'pattern': '4x2',
                "pruning_type": "snip_progressive",
                "pruning_frequency": 5
            }
        ]
        config = WeightPruningConfig(
            local_configs,
            target_sparsity=0.8,
            sparsity_decay_type="cos",
            excluded_op_names=["downsample.*"],
            pruning_scope="local",
            min_sparsity_ratio_per_op=0.1,
            start_step=1,
            end_step=10
        )
        prune = Pruning(config)
        prune.update_config(start_step=1, end_step=10)
        prune.model = self.model

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        x_train = np.random.uniform(low=0., high=1., size=tuple([10, 3, 224, 224]))
        y_train = np.random.randint(low=0, high=2, size=tuple([10]))
        x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
                                torch.tensor(y_train, dtype=torch.long)
        dummy_dataset = TensorDataset(x_train, y_train)
        dummy_dataloader = DataLoader(dummy_dataset)
        prune.on_train_begin()
        for epoch in range(5):
            self.model.train()
            prune.on_epoch_begin(epoch)
            local_step = 0
            for image, target in dummy_dataloader:
                prune.on_step_begin(local_step)
                output = self.model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                prune.on_before_optimizer_step()
                optimizer.step()
                prune.on_after_optimizer_step()
                prune.on_step_end()
                local_step += 1

            prune.on_epoch_end()
        prune.get_sparsity_ratio()
        prune.on_train_end()
        prune.on_before_eval()
        prune.on_after_eval()

    def test_utils(self):
        from intel_extension_for_transformers.transformers.utils.utility import remove_label

        dataset = remove_label({'labels': [], 'ids': []})
        dataset = remove_label({'start_positions': [], 'end_positions': [], 'ids': []})


    def test_sparsity_config_loading(self):
        config = SparsityConfig.from_pretrained("Intel/gpt-j-6b-sparse")
        config.save_pretrained("sparsity_config_dir")
        loaded_config = SparsityConfig.from_pretrained("sparsity_config_dir")
        self.assertEqual(config.sparse_pattern, loaded_config.sparse_pattern)
        self.assertEqual(config.dense_layers, loaded_config.dense_layers)



if __name__ == "__main__":
    unittest.main()
