import shutil
import numpy as np
import unittest
import tensorflow as tf
from datasets import load_dataset, load_metric
from transformers import (TFAutoModelForSequenceClassification, AutoTokenizer,
                          HfArgumentParser, TFTrainingArguments, set_seed,
                          DefaultDataCollator)
from intel_extension_for_transformers.optimization import (DistillationConfig, metrics)
from intel_extension_for_transformers.optimization.distillation import Criterion
from intel_extension_for_transformers.optimization.optimizer_tf import TFOptimization


class TestDistillation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
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

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        shutil.rmtree('./distilled_model', ignore_errors=True)

    def test_tf_model_distil(self):
        metric = load_metric("glue", "sst2")
        def compute_metrics(preds, label_ids):
            preds = preds["logits"]
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

        self.optimizer = TFOptimization(model=self.model,
                                        args=self.args,
                                        train_dataset=self.dummy_dataset,
                                        compute_metrics=compute_metrics)
        metric_ = metrics.Metric(name="eval_accuracy")
        # 'CrossEntropyLoss', 'SparseCategoricalCrossentropy', 'KnowledgeDistillationLoss'
        criterion = Criterion(name='KnowledgeLoss',
                              layer_mappings=[['classifier', 'classifier']],
                              loss_types=['CE', 'CE'],
                              loss_weight_ratio=[0.5, 0.5],
                              add_origin_loss=False)
        distillation_conf = DistillationConfig(metrics=metric_,
                                               criterion=criterion)
        def eval_func(model):
            return 1
        distilled_model = self.optimizer.distill(
            distillation_config=distillation_conf,
            teacher_model=self.teacher_model,
            eval_func=eval_func,
            train_func=self.optimizer.build_train_func
        )
        distilled_model = self.optimizer.distill(
            distillation_config=distillation_conf,
            teacher_model=self.teacher_model,
            eval_func=None,
            train_func=None
        )
        # distilled_weight = copy.deepcopy(distilled_model.model.classifier.get_weights())


if __name__ == "__main__":
    unittest.main()
