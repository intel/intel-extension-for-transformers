import os
import shutil
import unittest
from nlp_toolkit import (
    metrics,
    objectives,
    QuantizationConfig,
)
from nlp_toolkit.optimization.benchmark import (
    PyTorchBenchmark,
    PyTorchBenchmarkArguments,
    ExecutorBenchmark,
    ExecutorBenchmarkArguments,
)
from transformers import (
    AutoConfig,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ["GLOG_minloglevel"] = "2"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from test_quantization import TestQuantization
        TestQuantization.setUpClass()
        self.trainer = TestQuantization.trainer
        self.trainer.export_to_onnx('fp32.onnx')
        tune_metric = metrics.Metric(
                name="eval_loss", greater_is_better=False, 
                is_relative=False, criterion=0.5
            )
        quantization_config = QuantizationConfig(
            approach='PostTrainingStatic',
            metrics=[tune_metric],
            objectives=[objectives.performance]
        )
        self.trainer.quantize(quant_config=quantization_config, provider="inc")
        self.trainer.enable_executor = True
        self.trainer.export_to_onnx('int8.onnx')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('fp32.onnx', ignore_errors=True)
        shutil.rmtree('int8.onnx', ignore_errors=True)

    def check_results_dict_not_empty(self, results):
        for model_result in results.values():
            for batch_size, sequence_length in zip(model_result["bs"], model_result["ss"]):
                result = model_result["result"][batch_size][sequence_length]
                self.assertIsNotNone(result)

    def test_benchmark(self):
        MODEL_ID_FP32 = "distilbert-base-uncased-finetuned-sst-2-english"
        MODEL_ID_INT8 = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
        benchmark_args = PyTorchBenchmarkArguments(
            models=[MODEL_ID_FP32, MODEL_ID_INT8],
            training=False,
            memory=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
            only_pretrain_model=True,
        )
        benchmark = PyTorchBenchmark(benchmark_args)
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)

    def test_torchscript_benchmark(self):
        MODEL_ID_FP32 = "distilbert-base-uncased-finetuned-sst-2-english"
        MODEL_ID_INT8 = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
        benchmark_args = PyTorchBenchmarkArguments(
            models=[MODEL_ID_FP32, MODEL_ID_INT8],
            training=False,
            memory=False,
            inference=True,
            torchscript=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
            only_pretrain_model=True,
        )
        benchmark = PyTorchBenchmark(benchmark_args)
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)

    def test_executor_benchmark(self):
        MODEL_ID_FP32 = 'fp32.onnx'
        MODEL_ID_INT8 = 'int8.onnx'
        config = AutoConfig.from_pretrained(MODEL_NAME)
        benchmark_args = ExecutorBenchmarkArguments(
            models=[MODEL_ID_FP32, MODEL_ID_INT8],
            memory=False,
            inference=True,
            sequence_lengths=[32],
            batch_sizes=[1],
            multi_process=False,
            only_pretrain_model=True,
        )
        benchmark = ExecutorBenchmark(benchmark_args, configs=[config, config])
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)


if __name__ == "__main__":
    unittest.main()
