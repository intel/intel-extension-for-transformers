from intel_extension_for_transformers.optimization.utils.utility_tf import get_filepath
import numpy as np
import os
import shutil
import tensorflow as tf 
import unittest
from datasets import load_dataset, load_metric
from intel_extension_for_transformers.optimization import (
    metrics,
    PrunerConfig,
    PruningConfig,
    TFOptimization
)
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    HfArgumentParser,
    TFAutoModelForSequenceClassification,
    TFTrainingArguments,
)

os.environ["WANDB_DISABLED"] = "true"


class TestTFPruning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            'bhadresh-savani/distilbert-base-uncased-sentiment-sst2'
        )
        raw_datasets = load_dataset("glue", "sst2")["validation"]
        tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-sentiment-sst2")
        non_label_column_names = [name for name in raw_datasets.column_names if name != "label"]
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples["sentence"],)
            )
            result = tokenizer(*args, padding=True, max_length=64, truncation=True)

            return result
        raw_datasets = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=False)
        data_collator = DefaultDataCollator(return_tensors="tf")
        dataset = raw_datasets.select(range(10))
        self.dummy_dataset = dataset.to_tf_dataset(
            columns=[col for col in dataset.column_names if col not in 
                     set(non_label_column_names + ["label"])],
            shuffle=False,
            batch_size=2,
            collate_fn=data_collator,
            drop_remainder=False,
            # `label_cols` is needed for user-defined losses, such as in this example
            # datasets v2.3.x need "labels", not "label"
            label_cols=["labels"] if "label" in dataset.column_names else None,
        )
        parser = HfArgumentParser(TFTrainingArguments)
        self.args = parser.parse_args_into_dataclasses(args=["--output_dir", "./quantized_model",
                                                        "--per_device_eval_batch_size", "2"])[0]
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.args.learning_rate,
            beta_1=self.args.adam_beta1,
            beta_2=self.args.adam_beta2,
            epsilon=self.args.adam_epsilon,
            clipnorm=self.args.max_grad_norm,
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        )
        metrics = ["accuracy"]
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        shutil.rmtree('./quantized_model', ignore_errors=True)

    def test_tf_model_quant(self):
        # check whether it is possible to set distributed environment
        # only for coverage currently
        from intel_extension_for_transformers.optimization.utils.utility_tf import distributed_init
        distributed_init(["localhost:12345","localhost:23456"], "worker", 0)
        self.assertTrue(os.environ['TF_CONFIG'] != None)
        del os.environ['TF_CONFIG']
        # check whether filepath can be set correctly if using distributed environment
        # only for coverage currently
        from intel_extension_for_transformers.optimization.utils.utility_tf import get_filepath
        self.assertTrue(type(get_filepath("dummy", "worker", 0)) == str)
        self.assertTrue(type(get_filepath("dummy", "worker", 1)) == str)
        self.assertTrue(get_filepath("dummy", "worker", 0) != get_filepath("dummy", "worker", 1))

        metric = load_metric("glue", "sst2")
        def compute_metrics(preds, label_ids):
            preds = preds["logits"]
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        self.optimizer = TFOptimization(
            model=self.model,
            args=self.args,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
            compute_metrics=compute_metrics,
        )
        tune_metric = metrics.Metric(
            name="accuracy", greater_is_better=True, is_relative=True, criterion=0.01,
        )
        prune_type = 'BasicMagnitude'
        target_sparsity_ratio = 0.1
        pruner_config = PrunerConfig(prune_type=prune_type, target_sparsity_ratio=target_sparsity_ratio)
        pruning_conf = PruningConfig(
            epochs=int(1), pruner_config=pruner_config, metrics=tune_metric
        )
        p_model = self.optimizer.prune(pruning_config=pruning_conf)
        loaded_model = tf.saved_model.load(self.args.output_dir)
        p_model = self.optimizer.prune(pruning_config=pruning_conf,
                                train_dataset=self.dummy_dataset,
                                eval_dataset=self.dummy_dataset,)

        def eval_func(model):
            return 1

        def train_func(model):
            return model
        
        self.optimizer.prune(pruning_config=pruning_conf,
                             train_func=train_func, 
                             eval_func=eval_func)


if __name__ == "__main__":
    unittest.main()
