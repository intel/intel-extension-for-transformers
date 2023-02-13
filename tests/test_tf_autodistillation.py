import shutil
import numpy as np
import unittest
import tensorflow as tf
from datasets import load_dataset, load_metric
from transformers import (TFAutoModelForSequenceClassification, AutoTokenizer,
                          DefaultDataCollator, HfArgumentParser,
                          TFTrainingArguments, set_seed)
from intel_extension_for_transformers.optimization import (
    AutoDistillationConfig,
    TFDistillationConfig,
    metrics,
)
from intel_extension_for_transformers.optimization.optimizer_tf import TFOptimization
from intel_extension_for_transformers.optimization.utils.utility_tf import distributed_init

def compute_metrics(preds, label_ids):
    metric = load_metric("glue", "sst2")
    preds = preds["logits"]
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

class TestAutoDistillation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
        set_seed(42)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased')
        self.teacher_model = TFAutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english')

        raw_datasets = load_dataset("glue", "sst2")["validation"]
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        non_label_column_names = [
            name for name in raw_datasets.column_names if name != "label"
        ]

        def preprocess_function(examples):
            # Tokenize the texts
            args = ((examples['sentence'], ))
            result = self.tokenizer(*args,
                                    padding=True,
                                    max_length=64,
                                    truncation=True)
            return result

        raw_datasets = raw_datasets.map(preprocess_function,
                                        batched=True,
                                        load_from_cache_file=False)
        data_collator = DefaultDataCollator(return_tensors="tf")
        dataset = raw_datasets.select(range(10))
        self.dummy_dataset = dataset.to_tf_dataset(
            columns=[
                col for col in dataset.column_names
                if col not in set(non_label_column_names + ["label"])
            ],
            shuffle=False,
            batch_size=2,
            collate_fn=data_collator,
            drop_remainder=False,
            # `label_cols` is needed for user-defined losses, such as in this example
            # datasets v2.3.x need "labels", not "label"
            label_cols=["labels"]
            if "label" in dataset.column_names else None,
        )
        parser = HfArgumentParser(TFTrainingArguments)
        self.args = parser.parse_args_into_dataclasses(args=[
            "--output_dir", "./distilled_model",
            "--per_device_eval_batch_size", "2"
        ])[0]
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.args.learning_rate,
            beta_1=self.args.adam_beta1,
            beta_2=self.args.adam_beta2,
            epsilon=self.args.adam_epsilon,
            clipnorm=self.args.max_grad_norm,
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        metrics = ["accuracy"]
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        self.optimizer = TFOptimization(model=self.model,
                                        args=self.args,
                                        train_dataset=self.dummy_dataset,
                                        eval_dataset=self.dummy_dataset,
                                        compute_metrics=compute_metrics)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        shutil.rmtree('./distilled_model', ignore_errors=True)

    def test_tf_auto_distillation(self):
        for search_algorithm in ['BO', 'Grid', 'Random']:
            max_trials = 6 if search_algorithm == 'Random' else 3
            autodistillation_config = AutoDistillationConfig(
                search_space={
                    'hidden_size': [120, 240],
                    'intermediate_size': [256, 512]
                },
                search_algorithm=search_algorithm,
                max_trials=max_trials,
                metrics=[
                    metrics.Metric(name="metric", greater_is_better=False)
                ],
                knowledge_transfer=TFDistillationConfig(
                    train_steps=[3],
                    loss_types=['CE', 'CE'],
                    loss_weights=[0.5, 0.5],
                    temperature=1.0
                ),
                regular_distillation=TFDistillationConfig(
                    train_steps=[3],
                    loss_types=['CE', 'CE'],
                    loss_weights=[0.5, 0.5],
                    temperature=1.0
                )
            )
            best_model_archs1 = self.optimizer.autodistill(
                autodistillation_config,
                self.teacher_model,
                model_cls=TFAutoModelForSequenceClassification,
                train_func=None,
                eval_func=None
            )

            best_model_archs2 = self.optimizer.autodistill(
                autodistillation_config,
                self.teacher_model,
                model_cls=TFAutoModelForSequenceClassification,
                train_func=self.optimizer.build_train_func,
                eval_func=self.optimizer.builtin_eval_func
            )
            # check best model architectures
            self.assertTrue(len(best_model_archs1) > 0)
            self.assertTrue(len(best_model_archs2) > 0)


if __name__ == "__main__":
    unittest.main()
