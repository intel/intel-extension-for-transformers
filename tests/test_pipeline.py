import os
import unittest
from nlp_toolkit.optimization.pipeline import pipeline
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ["GLOG_minloglevel"] = "2"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

message = "The output scores should be close to 0.9999."


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from test_benchmark import TestBenchmark
        TestBenchmark.setUpClass()
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    @classmethod
    def tearDownClass(self):
        from test_benchmark import TestBenchmark
        TestBenchmark.tearDownClass()

    def test_fp32_pt_model(self):
        text_classifier = pipeline(
            task="text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",
        )
        outputs = text_classifier("This is great !")
        self.assertAlmostEqual(outputs[0]['score'], 0.9999, None, message, 0.0001)

    def test_int8_pt_model(self):
        import torch
        text_classifier = pipeline(
            task="text-classification",
            model="Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static",
            framework="pt",
        )
        outputs = text_classifier("This is great !")
        self.assertAlmostEqual(outputs[0]['score'], 0.9999, None, message, 0.0001)

    def test_fp32_executor_model(self):
        text_classifier = pipeline(
            task="text-classification",
            config=self.config,
            tokenizer=self.tokenizer,
            model='fp32.onnx',
            model_kwargs={'backend': "executor"},
        )
        outputs = text_classifier(
            "But believe it or not , it 's one of the most "
            "beautiful , evocative works I 've seen ."
        )
        self.assertAlmostEqual(outputs[0]['score'], 0.9999, None, message, 0.0001)

    def test_int8_executor_model(self):
        text_classifier = pipeline(
            task="text-classification",
            config=self.config,
            tokenizer=self.tokenizer,
            model='int8.onnx',
            model_kwargs={'backend': "executor"},
        )
        outputs = text_classifier(
            "But believe it or not , it 's one of the most "
            "beautiful , evocative works I 've seen ."
        )
        self.assertAlmostEqual(outputs[0]['score'], 0.9999, None, message, 0.0001)


if __name__ == "__main__":
    unittest.main()
