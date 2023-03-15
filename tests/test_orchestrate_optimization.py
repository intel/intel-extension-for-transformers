import copy
import torch
import numpy as np
import os
import shutil
import torch.utils.data as data
import unittest
from datasets import load_dataset, load_metric
from intel_extension_for_transformers.optimization import (
    PrunerConfig,
    PruningConfig,
    DistillationConfig,
    QuantizationConfig,
    DistillationCriterionMode,
    metrics,
    objectives,
    OptimizedModel,
)
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from intel_extension_for_transformers.optimization.distillation import Criterion

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

os.environ["WANDB_DISABLED"] = "true"

class TestOrchestrateOptimizations(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        set_seed(42)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased'
        )
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english'
        )
        raw_datasets = load_dataset("glue", "sst2")["validation"]
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples['sentence'],)
            )
            result = tokenizer(*args, padding=True, max_length=64, truncation=True)
            return result
        raw_datasets = raw_datasets.map(
            preprocess_function, batched=True, load_from_cache_file=True
        )
        eval_dataset = raw_datasets.select(range(30))
        self.dataset = eval_dataset

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./orchestrate_optimizations_model', ignore_errors=True)

    def test_fx_orchestrate_optimization(self):
        metric = load_metric("accuracy")
        def compute_metrics(p):
            preds = p.predictions
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)
        origin_weight = copy.deepcopy(self.model.classifier.weight)
        for mode in DistillationCriterionMode:
            print("Distillation approach:", mode.value)
            self.trainer = NLPTrainer(
                model=copy.deepcopy(self.model),
                train_dataset=self.dataset,
                eval_dataset=self.dataset,
                compute_metrics=compute_metrics,
            )
            self.trainer.calib_dataloader = self.trainer.get_eval_dataloader()
        tune_metric = metrics.Metric(
            name="eval_accuracy", is_relative=True, criterion=0.5
        )
        pruner_config = PrunerConfig(prune_type='PatternLock', target_sparsity_ratio=0.9)
        pruning_conf = PruningConfig(framework="pytorch_fx",pruner_config=[pruner_config], metrics=tune_metric)
        distillation_conf = DistillationConfig(framework="pytorch_fx", metrics=tune_metric)

        objective = objectives.performance
        quantization_conf = QuantizationConfig(
            approach="QuantizationAwareTraining",
            max_trials=600,
            metrics=[tune_metric],
            objectives=[objective]
        )

        from neural_compressor.adaptor.torch_utils.symbolic_trace import symbolic_trace
        self.model = symbolic_trace(self.model, is_qat=True)
        self.trainer.model = self.model
        conf_list = [pruning_conf, distillation_conf, quantization_conf]
        opt_model = self.trainer.orchestrate_optimizations(config_list=conf_list, teacher_model=self.teacher_model)
        self.assertTrue("quantize" in str(type(opt_model.classifier.module)))


if __name__ == "__main__":
    unittest.main()
