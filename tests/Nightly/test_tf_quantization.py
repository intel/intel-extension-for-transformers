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

import numpy as np
import os
import shutil
import tensorflow as tf
import unittest
from datasets import load_dataset, load_metric
from intel_extension_for_transformers.transformers import (
    metrics,
    objectives,
    QuantizationConfig,
    TFOptimization
)
# from intel_extension_for_transformers.transformers import metrics, objectives
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    HfArgumentParser,
    TFAutoModelForSequenceClassification,
    TFTrainingArguments,
)

os.environ["WANDB_DISABLED"] = "true"


class TestTFQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            'hf-internal-testing/tiny-random-DistilBertForSequenceClassification'
        )
        raw_datasets = load_dataset("glue", "sst2")["validation"]
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-DistilBertForSequenceClassification")
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


    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        shutil.rmtree('./quantized_model', ignore_errors=True)

    def test_tf_model_quant(self):
        parser = HfArgumentParser(TFTrainingArguments)
        args = parser.parse_args_into_dataclasses(args=["--output_dir", "./quantized_model",
                                                        "--per_device_eval_batch_size", "2"])
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
            args=args[0],
            compute_metrics=compute_metrics
        )
        tune_metric = metrics.Metric(
            name="accuracy", greater_is_better=True, is_relative=False, criterion=0.5
        )
        quantization_config = QuantizationConfig(
            framework="tensorflow",
            approach="POSTTRAININGSTATIC",
            metrics=[tune_metric],
            objectives=[objectives.performance]
        )
        quantized_model = self.optimizer.quantize(quant_config=quantization_config,
            train_dataset=self.dummy_dataset, eval_dataset=self.dummy_dataset)
        loaded_model = tf.saved_model.load(args[0].output_dir)

        def eval_func(model):
            return 1

        def train_func(model):
            return model

        self.optimizer.quantize(quant_config=quantization_config,
                                train_func=train_func,
                                eval_func=eval_func)

        quantization_config = QuantizationConfig(
            framework="tensorflow",
            approach="POSTTRAININGSTATIC",
            metrics=[tune_metric],
            objectives=[objectives.performance],
            recipes={"first_conv_or_matmul_quantization": True,
                     "last_conv_or_matmul_quantization": True,
                     }
        )
        self.optimizer.quantize(quant_config=quantization_config,
                                train_func=train_func,
                                eval_func=eval_func)


if __name__ == "__main__":
    unittest.main()
