import os
import shutil
import unittest

from nlp_toolkit import (
    DistillationConfig,
    metrics,
    objectives,
    Pruner,
    PruningConfig,
    QuantizationConfig,
)
from nlp_toolkit.optimization.distillation import Criterion as DistillationCriterion, \
                                                  DistillationCriterionMode


class CustomPruner():
    def __init__(self, start_epoch=None, end_epoch=None, initial_sparsity=None,
                 target_sparsity_ratio=None, update_frequency=1, prune_type='BasicMagnitude',
                 method='per_tensor', names=[], parameters=None):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.update_frequency = update_frequency
        self.target_sparsity_ratio = target_sparsity_ratio
        self.initial_sparsity = initial_sparsity
        self.update_frequency = update_frequency


class TestConfig(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)

    def test_quantization_config_with_init(self):
        metric1 = metrics.Metric(
            name="F1", greater_is_better=False, is_relative=False, criterion=0.02, weight_ratio=0.5
        )
        metric2 = metrics.Metric(
            name="accuracy", greater_is_better=False, is_relative=False,
            criterion=0.02, weight_ratio=0.5
        )
        objective1 = objectives.performance
        objective2 = objectives.modelsize
        quantization_config = QuantizationConfig(
                                framework="pytorch",
                                approach="PostTrainingDynamic",
                                timeout=600,
                                max_trials=300,
                                metrics=[metric1, metric2],
                                objectives=[objective1, objective2],
                            )

        self.assertEqual(quantization_config.approach, "post_training_dynamic_quant")
        self.assertEqual(quantization_config.metrics[0].criterion, 0.02)
        self.assertEqual(quantization_config.objectives[1].name, "modelsize")
        self.assertEqual(quantization_config.timeout, 600)
        self.assertEqual(quantization_config.max_trials, 300)

    def test_quantization_config(self):
        quantization_config = QuantizationConfig()
        quantization_config.approach = "PostTrainingStatic"
        quantization_config.framework = "pytorch"
        metric = metrics.Metric(name="F1", greater_is_better=False, criterion=0.02, is_relative=True)
        quantization_config.metrics = metric
        objective1 = objectives.Objective(name="performance", greater_is_better=True)
        objective2 = objectives.Objective(name="modelsize", greater_is_better=False)
        quantization_config.objectives = [objective1, objective2]

        quantization_config.timeout = 600
        quantization_config.max_trials = 300
        quantization_config.output_dir = "./savedresult"

        self.assertEqual(quantization_config.approach, "post_training_static_quant")
        self.assertEqual(quantization_config.metrics.criterion, 0.02)
        self.assertEqual(quantization_config.objectives[1].name, "modelsize")
        self.assertEqual(quantization_config.timeout, 600)
        self.assertEqual(quantization_config.max_trials, 300)
        self.assertEqual(quantization_config.output_dir, "./savedresult")

    def test_pruning_config(self):
        pruning_config = PruningConfig()
        pruner = Pruner()
        metric = metrics.Metric(name="F1")
        pruning_config.pruner = pruner
        pruning_config.framework = "pytorch"
        pruning_config.target_sparsity_ratio = 0.1
        pruning_config.epoch_range = [0, 4]
        pruning_config.metrics = [metric]

        self.assertEqual(pruning_config.pruner, [pruner])
        self.assertEqual(pruning_config.framework, "pytorch")
        self.assertEqual(pruning_config.target_sparsity_ratio, 0.1)
        self.assertEqual(pruning_config.epoch_range, [0, 4])
        self.assertEqual(pruning_config.metrics, [metric])

    def test_distillation_config(self):
        metric1 = metrics.Metric(name="F1")
        metric2 = metrics.Metric(name="accuracy")
        criterion = DistillationCriterion(
            name="KnowledgeLoss",
            temperature=1.0,
            loss_types=["CE", "KL"],
            loss_weight_ratio=[0, 1]
        )
        distillation_config = DistillationConfig(
            framework="pytorch",
            criterion=criterion,
            metrics=[metric1, metric2]
        )

        self.assertEqual(distillation_config.framework, "pytorch")
        self.assertEqual(list(distillation_config.criterion.keys())[0],
                         DistillationCriterionMode[criterion.name.upper()].value)
        self.assertEqual(distillation_config.metrics, [metric1, metric2])


if __name__ == "__main__":
    unittest.main()
