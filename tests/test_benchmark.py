import os
import shutil
import unittest
import neural_compressor.adaptor.pytorch as nc_torch
from distutils.version import LooseVersion
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

PT_VERSION = nc_torch.get_torch_version()
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class TestBenchmark(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)

    @classmethod
    def check_results_dict_not_empty(cls, results):
        for model_result in results.values():
            for batch_size, sequence_length in zip(model_result["bs"], model_result["ss"]):
                result = model_result["result"][batch_size][sequence_length]
                cls.assertIsNotNone(cls, result)

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
        TestBenchmark.check_results_dict_not_empty(results.time_inference_result)

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
        TestBenchmark.check_results_dict_not_empty(results.time_inference_result)


@unittest.skipIf(PT_VERSION >= LooseVersion("1.12.0"),
    "Please use PyTroch 1.11 or lower version for executor backend")
class TestExecutorBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_quantization import TestQuantization
        TestQuantization.setUpClass()
        cls.trainer = TestQuantization.trainer
        # By default, the onnx model is saved in tmp_trainer dir
        cls.trainer.export_to_onnx()
        tune_metric = metrics.Metric(
                name="eval_loss", greater_is_better=False, 
                is_relative=False, criterion=0.5
            )
        quantization_config = QuantizationConfig(
            approach='PostTrainingStatic',
            metrics=[tune_metric],
            objectives=[objectives.performance]
        )
        cls.trainer.quantize(quant_config=quantization_config, provider="inc")
        cls.trainer.enable_executor = True
        cls.trainer.export_to_onnx()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./nc_workspace', ignore_errors=True)
        if os.path.exists('augmented_model.onnx'):
            os.remove('augmented_model.onnx')

    def test_executor_benchmark(self):
        MODEL_ID_FP32 = 'tmp_trainer/fp32-model.onnx'
        MODEL_ID_INT8 = 'tmp_trainer/int8-model.onnx'
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
        TestBenchmark.check_results_dict_not_empty(results.time_inference_result)


if __name__ == "__main__":
    unittest.main()
