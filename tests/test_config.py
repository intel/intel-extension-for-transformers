import os
import shutil
import unittest

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
)

from nlp_toolkit.optimization.trainer import NLPTrainer
from nlp_toolkit.optimization.config import QuantizationConfig, PruningConfig, DistillationConfig, OptimizeConfig


class CustomPruner():
    def __init__(self, start_epoch=None, end_epoch=None, initial_sparsity=None,
                 target_sparsity=None, update_frequency=1, prune_type='basic_magnitude',
                 method='per_tensor', names=[], parameters=None):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.update_frequency = update_frequency
        self.target_sparsity = target_sparsity
        self.initial_sparsity = initial_sparsity
        self.update_frequency = update_frequency

metrics = {
            "metrics":["eval_accuracy", "eval_f1"],
            "weights": [0.5,0.5],
            "Higher_is_better": [True, True]
          }
objectives = {
                "objectives":["performance", "accuracy"],
                "weights": [0.2, 0.8]
             }

class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased'
        )
        self.trainer = NLPTrainer(
            model=self.model
        )
        self.trainer.args.output_dir='./tmp_trainer'

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)

    def test_quantization_config(self):
        quantization_config = QuantizationConfig()
        quantization_config.approach = "post_training_static_quant"
        quantization_config.metric_tolerance = {'relative': 0.02, 'higher_is_better': True}
        with self.assertRaises(TypeError):
            quantization_config.metric_tolerance = {'relative': '0.01', 'higher_is_better': True}
        with self.assertRaises(ValueError):
            quantization_config.metric_tolerance = {'relative': 1.50, 'higher_is_better': True}
        with self.assertRaises(AssertionError):
            quantization_config.metric_tolerance = 0.01

        quantization_config.framework = "pytorch"
        quantization_config.stategy = "basic"
        quantization_config.objective = "performance"
        quantization_config.timeout = 600
        quantization_config.max_trials = 300
        quantization_config.performance_only = True
        quantization_config.random_seed = 9527
        quantization_config.tensorboard = True
        quantization_config.save_path = "./savedresult"
        quantization_config.resume_path = "./nc_workspace"
        quantization_config.metrics = metrics
        quantization_config.objectives = objectives 

        self.trainer.quantization = quantization_config
        self.assertEqual(self.trainer.quantization.approach,"post_training_static_quant")
        self.assertEqual(self.trainer.quantization.metric_tolerance.relative, 0.02)
        self.assertEqual(self.trainer.quantization.framework, "pytorch")
        self.assertEqual(self.trainer.quantization.stategy, "basic")
        self.assertEqual(self.trainer.quantization.objective, "performance")
        self.assertEqual(self.trainer.quantization.timeout, 600)
        self.assertEqual(self.trainer.quantization.max_trials, 300)
        self.assertEqual(self.trainer.quantization.performance_only, True)
        self.assertEqual(self.trainer.quantization.random_seed, 9527)
        self.assertEqual(self.trainer.quantization.tensorboard, True)
        self.assertEqual(self.trainer.quantization.save_path, "./savedresult")
        self.assertEqual(self.trainer.quantization.resume_path, "./nc_workspace")
        self.assertEqual(self.trainer.quantization.metrics, metrics)
        self.assertEqual(self.trainer.quantization.objectives, objectives)
    
    def test_pruning_config(self):
        pruning_config = PruningConfig()
        pruning_config.custom_pruner = CustomPruner
        pruning_config.approach = "basic_magnitude"
        pruning_config.framework = "pytorch"
        pruning_config.target_sparsity = 0.1
        pruning_config.epoch_range = [0,4]
        pruning_config.metrics = metrics
        
        self.trainer.pruning = pruning_config
        self.assertEqual(self.trainer.pruning.custom_pruner, [CustomPruner])
        self.assertEqual(self.trainer.pruning.approach, "basic_magnitude")
        self.assertEqual(self.trainer.pruning.framework, "pytorch")
        self.assertEqual(self.trainer.pruning.target_sparsity, 0.1)
        self.assertEqual(self.trainer.pruning.epoch_range, [0,4])
        self.assertEqual(self.trainer.pruning.metrics, metrics)

    def test_distillation_config(self):
        distillation_config = DistillationConfig()
        distillation_config.framework = "pytorch"
        distillation_criterion = {
                    "KnowledgeDistillationLoss" :{
                        "temperature": 1.0, 
                        "loss_types":["CE", "KL"],
                        "loss_weights":[0,1]
                        }
                    }
        distillation_config.criterion = distillation_criterion
        distillation_config.metrics = metrics
        
        self.trainer.distillation = distillation_config
        self.assertEqual(self.trainer.distillation.framework, "pytorch")
        self.assertEqual(self.trainer.distillation.criterion, distillation_criterion)
        self.assertEqual(self.trainer.distillation.metrics, metrics)
    
    def test_optimize_config(self):
        optimize_config = OptimizeConfig()
        optimize_config.provider_arguments = {
            "framework": "pytorch",
            "quantization":{
                "approach": "PostTrainingStatic",
                "criterion": {"relative": 0.001},
                "strategy": "basic",
                "timeout": 600,
                "max_trials": 300,
                "performance_only": False,
                "save_path": './savedresult',
                "metrics":{
                    "metrics":["eval_accuracy", "eval_f1"],
                    "weights": [0.5,0.5],
                    "Higher_is_better": [True, True]
                    },
                "objectives": {
                    "objectives":["performance", "accuracy"],
                    "weights": [0.2, 0.8]
                    }
            },
            "pruning":{
                "approach": "magnitude",
                "custom_pruner": CustomPruner,
                "target_sparsity": 0.1,
                "epoch_range": [0,3],
                "metrics":{
                    "metrics":["eval_accuracy", "eval_f1"],
                    "weights": [0.5,0.5],
                    "Higher_is_better": [True, True]
                    }
            },
            "distillation":{
                "criterion": {
                    "KnowledgeDistillationLoss" :{
                        "temperature": 1.0, 
                        "loss_types":["CE", "KL"],
                        "loss_weights":[0,1]
                        }
                    },
                "metrics":{
                    "metrics":["eval_accuracy", "eval_f1"],
                    "weights": [0.5,0.5],
                    "Higher_is_better": [True, True]
                    }
            }

        }

        optimize_config.parse_inc_arguments()
        with self.assertRaises(AssertionError):
            optimize_config.parse_nncf_arguments()


if __name__ == "__main__":
    unittest.main()
