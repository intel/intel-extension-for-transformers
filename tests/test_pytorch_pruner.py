import os
import sys
import unittest

sys.path.insert(0, './')

import torch
import numpy as np
# from neural_compressor.config import WeightPruningConfig
# from intel_extension_for_transformers.optimization.pruner.pruning import Pruning
from intel_extension_for_transformers.optimization.pruner import WeightPruningConfig, Pruning
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



local_schedulers_config = [
    {
        "start_step": 0,
        "end_step": 2,
        "pruning_type": "magnitude",
        "op_names": ['layer1.*'],
        "excluded_op_names": ['layer2.*'],
        "pruning_scope": "global",
        "target_sparsity": 0.5,
        "pattern": "4x1"
    },
    {
        "start_step": 1,
        "end_step": 10,
        "target_sparsity": 0.5,
        "pruning_type": "snip_momentum",
        "pruning_frequency": 2,
        "op_names": ['layer2.*'],
        "pruning_scope": "local",
        "target_sparsity": 0.75,
        "pattern": "32x1",
        "sparsity_decay_type": "exp"
    },
    {
        "start_step": 1,
        "end_step": 10,
        "pruning_type": "snip_progressive",
        "pruning_frequency": 2,
        "op_names": ['layer2.*'],
        "pruning_scope": "local",
        "target_sparsity": 0.7,
        "pattern": "4x2",
        "sparsity_decay_type": "linear"
    }
]

fake_snip_config = WeightPruningConfig(local_schedulers_config, target_sparsity=0.9, start_step=0, \
                                       end_step=10, pruning_frequency=1, sparsity_decay_type="exp")

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
                "false_key": "this is to test unsupport keys"
            },
            {
                "op_names": ['layer2.*'],
                'target_sparsity': 0.5,
                'pattern': '2:4'
            },
            {
                "op_names": ['layer3.*'],
                'target_sparsity': 0.7,
                'pattern': '4x1',
                "pruning_type": "snip_progressive"
            },
            {
                "start_step": 2,
                "end_step": 8,
                "pruning_type": "gradient",
                "pruning_frequency": 2,
                "op_names": ['fc'],
                "pruning_scope": "local",
                "target_sparsity": 0.75,
                "pattern": "1x1",
                "sparsity_decay_type": "cube",
                "reg_type": "group_lasso",
                "parameters": {'reg_coeff': 0.0}
            }
        ]
        config = WeightPruningConfig(
            local_configs,
            target_sparsity=0.8
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
            min_sparsity_ratio_per_op=0.1
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
        from intel_extension_for_transformers.optimization.utils.utility import remove_label

        dataset = remove_label({'labels': [], 'ids': []})
        dataset = remove_label({'start_positions': [], 'end_positions': [], 'ids': []})


if __name__ == "__main__":
    unittest.main()
