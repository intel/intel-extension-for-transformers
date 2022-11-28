import os
import sys
import unittest

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from intel_extension_for_transformers.optimization.pytorch_pruner.pruning import Pruning

sys.path.insert(0, './')

def build_fake_yaml_basic():
    fake_snip_yaml = """
    model:
      name: imagenet_prune
      framework: pytorch
    pruning:
      approach:
        weight_compression_pytorch:
          initial_sparsity: 0.0
          target_sparsity: 0.9
          max_sparsity_ratio_per_layer: 0.9
          start_step: 0
          end_step: 10
          excluded_names: ["classifier"]
          update_frequency_on_step: 1
          sparsity_decay_type: "exp"
          pruners:
            - !Pruner
                start_step: 0
                sparsity_decay_type: "cos"
                end_step: 10
                prune_type: "magnitude"
                names: ['layer1.*']
                extra_excluded_names: ['layer2.*']
                prune_domain: "global"
                pattern: "tile_pattern_4x1"
            - !Pruner
                start_step: 1
                end_step: 1
                target_sparsity: 0.5
                prune_type: "snip_momentum"
                update_frequency: 2
                names: ['layer2.*']
                prune_domain: local
                pattern: "tile_pattern_2:4"
            - !Pruner
                start_step: 2
                end_step: 8
                target_sparsity: 0.8
                prune_type: "snip"
                names: ['layer3.*']
                prune_domain: "local"
                pattern: "tile_pattern_16x1"
                sparsity_decay_type: "cube"
    """
    with open('fake_snip.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_snip_yaml)

def build_fake_yaml_channel():
    fake_channel_pruning_yaml = """
        model:
          name: imagenet_prune
          framework: pytorch
        pruning:
          approach:
            weight_compression_pytorch:
              initial_sparsity: 0.0
              target_sparsity: 0.9
              start_step: 0
              end_step: 10
              excluded_names: ["classifier"]
              update_frequency_on_step: 1
              sparsity_decay_type: "exp"
              pruners:
                - !Pruner
                    start_step: 5
                    end_step: 5
                    prune_type: "pattern_lock"
                    names: ['layer1.*']
                    extra_excluded_names: ['layer2.*']
                    prune_domain: "global"
                    pattern: "channelx1"
                - !Pruner
                    start_step: 1
                    end_step: 1
                    target_sparsity: 0.5
                    prune_type: "pattern_lock"
                    update_frequency: 2
                    names: ['layer2.*']
                    prune_domain: local
                    pattern: "2:4"
                - !Pruner
                    start_step: 2
                    end_step: 8
                    target_sparsity: 0.8
                    prune_type: "snip"
                    names: ['layer3.*']
                    prune_domain: "local"
                    pattern: "1xchannel"
                    sparsity_decay_type: "cube"
        """

    with open('fake_channel_pruning.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_channel_pruning_yaml)

class TestPytorchPruner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = torchvision.models.resnet18()
        build_fake_yaml_basic()
        build_fake_yaml_channel()
    
    @classmethod
    def tearDownClass(self):
        os.remove('fake_snip.yaml')
        os.remove('fake_channel_pruning.yaml')
    
    def test_pytorch_pruner(self):
        prune = Pruning('fake_snip.yaml')
        prune.update_items_for_all_pruners(start_step=1)
        prune.model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)

        # prepare data
        try:
            from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
            dummy_dataset = DummyDataset(shape=(10, 3, 224, 224), low=0., high=1., label=True)
        except:
            x_train = np.random.uniform(low=0., high=1., size=tuple([10, 3, 224, 224]))
            y_train = np.random.randint(low=0, high=2, size=tuple([10]))
            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
                                torch.tensor(y_train, dtype=torch.long)
            dummy_dataset = TensorDataset(x_train, y_train)
        dummy_dataloader = DataLoader(dummy_dataset)
        prune.on_train_begin()
        prune.update_items_for_all_pruners(update_frequency_on_step=1)
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
        r1, r2, r3 = prune.get_sparsity_ratio()
        prune.on_train_end()
        prune.on_before_eval()
        prune.on_after_eval()
        self.assertTrue(r1 > 0 and r2 > 0 and r3 > 0)
    
    def test_pytorch_pruner_channel_pruning(self):
        prune = Pruning("fake_channel_pruning.yaml")
        ##prune.generate_pruners()
        prune.model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)

        try:
            from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
            dummy_dataset = DummyDataset(shape=(10, 3, 224, 224), low=0., high=1., label=True)
        except:
            x_train = np.random.uniform(low=0., high=1., size=tuple([10, 3, 224, 224]))
            y_train = np.random.randint(low=0, high=2, size=tuple([10]))
            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
                                torch.tensor(y_train, dtype=torch.long)
            dummy_dataset = TensorDataset(x_train, y_train)
        dummy_dataloader = DataLoader(dummy_dataset)

        prune.on_train_begin()
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

if __name__ == "__main__":
    unittest.main()