import shutil
import unittest

from nlp_toolkit import (
    DistillationConfig,
    metrics,
    objectives,
    PrunerConfig,
    PruningConfig,
    QuantizationConfig,
    AutoDistillationConfig,
    FlashDistillationConfig,
)
from nlp_toolkit.optimization.distillation import Criterion as DistillationCriterion
from nlp_toolkit.optimization.distillation import DistillationCriterionMode
from nlp_toolkit.optimization.trainer import NLPTrainer

from transformers import (
    AutoModelForPreTraining,
)


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

        from neural_compressor.utils import constant
        quantization_config.op_wise = {
            'bert.encoder.layer.0.output.dense': constant.FP32,
        }
        quantization_config.resume_path = './saved_results'
        quantization_config.random_seed = 1
        quantization_config.strategy = 'basic'
        quantization_config.performance_only=True
        quantization_config.tensorboard=True
        self.assertTrue(isinstance(quantization_config.op_wise, dict))
        self.assertTrue(isinstance(quantization_config.strategy, str))
        self.assertEqual(quantization_config.random_seed, 1)
        self.assertEqual(quantization_config.strategy, 'basic')
        self.assertTrue(quantization_config.performance_only)
        self.assertTrue(quantization_config.tensorboard)

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
        pruner_config = PrunerConfig()
        metric = metrics.Metric(name="F1")
        pruning_config.pruner_config = pruner_config
        pruning_config.framework = "pytorch"
        pruning_config.target_sparsity_ratio = 0.1
        pruning_config.epoch_range = [0, 4]
        pruning_config.metrics = metric

        self.assertEqual(pruning_config.pruner_config, [pruner_config])
        self.assertEqual(pruning_config.framework, "pytorch")
        self.assertEqual(pruning_config.target_sparsity_ratio, 0.1)
        self.assertEqual(pruning_config.epoch_range, [0, 4])
        self.assertEqual(pruning_config.metrics, metric)

        pruning_config.pruner_config = [pruner_config]
        self.assertEqual(pruning_config.pruner_config, [pruner_config])

    def test_distillation_config(self):
        metric = metrics.Metric(name="eval_F1")
        criterion = DistillationCriterion(
            name="KnowledgeLoss",
            temperature=1.0,
            loss_types=["CE", "KL"],
            loss_weight_ratio=[0, 1]
        )
        distillation_config = DistillationConfig(
            framework="pytorch",
            criterion=criterion,
            metrics=metric
        )

        self.assertEqual(distillation_config.framework, "pytorch")
        self.assertEqual(list(distillation_config.criterion.keys())[0],
                         DistillationCriterionMode[criterion.name.upper()].value)
        self.assertEqual(distillation_config.metrics, metric)

        criterion = DistillationCriterion(
            name="InterMediateLayersloss",
            layer_mappings=[['classifier', 'classifier']],
            loss_types=['MSE'],
            loss_weight_ratio=[1.0],
            add_origin_loss=False
        )
        distillation_config = DistillationConfig(
            framework="pytorch",
            criterion=criterion,
            metrics=metric
        )

    def test_autodistillation_config(self):
        metric = [metrics.Metric(name="eval_loss", greater_is_better=False)]
        autodistillation_config = AutoDistillationConfig(
            search_space={'hidden_size': [128, 256]},
            metrics=metric,
            knowledge_transfer=FlashDistillationConfig(
                block_names=['mobilebert.encoder.layer.1'],
                layer_mappings_for_knowledge_transfer=[
                [('mobilebert.encoder.layer.1.output',
                    'bert.encoder.layer.1.output')]
                ],
                train_steps=[3]),
            regular_distillation=FlashDistillationConfig(
                layer_mappings_for_knowledge_transfer=[
                [('cls', '0', 'cls', '0')]
                ],
                loss_types=[['KL']],
                add_origin_loss=[True],
                train_steps=[5]
            ),
            max_trials=1,
            seed=1,
        )

        self.assertEqual(autodistillation_config.framework, "pytorch")
        self.assertEqual(autodistillation_config.search_algorithm, 'BO')
        self.assertEqual(autodistillation_config.max_trials, 1)
        self.assertEqual(autodistillation_config.seed, 1)
        self.assertEqual(autodistillation_config.metrics, metric)
        self.assertTrue(isinstance(autodistillation_config.search_space, dict))
        self.assertTrue(isinstance(autodistillation_config.knowledge_transfer, dict))
        self.assertTrue(isinstance(autodistillation_config.regular_distillation, dict))

    def test_trainer_config(self):
        model = AutoModelForPreTraining.from_pretrained(
            'google/bert_uncased_L-2_H-128_A-2'
        )
        trainer = NLPTrainer(model)
        trainer.resuming_checkpoint = 'saved_results'
        trainer.eval_func = None
        trainer.train_func = None
        trainer.calib_dataloader = None
        trainer.provider = 'inc'
        self.assertEqual(trainer.resuming_checkpoint, 'saved_results')
        self.assertEqual(trainer.eval_func, None)
        self.assertEqual(trainer.train_func, None)
        self.assertEqual(trainer.calib_dataloader, None)
        self.assertEqual(trainer.provider, 'inc')


if __name__ == "__main__":
    unittest.main()
