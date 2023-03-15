import os
import shutil
import unittest
# Import torch first and then transformers to avoid unresponsive jit.trace.
import torch
import transformers
from datasets import load_dataset
import neural_compressor.adaptor.pytorch as nc_torch
from intel_extension_for_transformers.optimization import BenchmarkConfig
from intel_extension_for_transformers.optimization.benchmark import benchmark


def get_example_inputs(model_name, dataset_name='sst2'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name, split='validation')
    text = dataset[0]['text'] if dataset_name=='lambada' else dataset[0]['sentence']
    example_inputs = tokenizer(text, padding='max_length', max_length=195, return_tensors='pt')
    return example_inputs

class TestBenchmark(unittest.TestCase):
    def test_fp32_model(self):
        config = BenchmarkConfig(
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=-1,
        )
        model_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/distilbert_base_uncased_sst2"):
            model_name_or_path = "/tf_dataset2/models/nlp_toolkit/distilbert_base_uncased_sst2"
        example_inputs = get_example_inputs(model_name_or_path)
        benchmark(model_name_or_path, config, example_inputs=example_inputs)

    def test_fp32_model_ipex(self):
        config = BenchmarkConfig(
            backend='ipex',
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=-1,
        )
        model_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/distilbert_base_uncased_sst2"):
            model_name_or_path = "/tf_dataset2/models/nlp_toolkit/distilbert_base_uncased_sst2"
        example_inputs = get_example_inputs(model_name_or_path)
        benchmark(model_name_or_path, config, example_inputs=example_inputs)

    def test_int8_model(self):
        config = BenchmarkConfig(
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=-1,
        )
        model_name_or_path = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/distilbert_sst2_int8"):
            model_name_or_path = "/tf_dataset2/models/nlp_toolkit/distilbert_sst2_int8"
        example_inputs = get_example_inputs(model_name_or_path)
        example_inputs = example_inputs['input_ids'] # for UT coverage
        benchmark(model_name_or_path, config, example_inputs=example_inputs)

    def test_int8_model_ipex(self):
        config = BenchmarkConfig(
            backend='ipex',
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=-1,
        )
        model_name_or_path = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/distilbert_sst2_int8"):
            model_name_or_path = "/tf_dataset2/models/nlp_toolkit/distilbert_sst2_int8"
        example_inputs = get_example_inputs(model_name_or_path)
        example_inputs = tuple(example_inputs.values()) # for UT coverage
        benchmark(model_name_or_path, config, example_inputs=example_inputs)

    def test_torchscript(self):
        config = BenchmarkConfig(
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=-1,
            torchscript=True,
        )
        model_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/distilbert_base_uncased_sst2"):
            model_name_or_path = "/tf_dataset2/models/nlp_toolkit/distilbert_base_uncased_sst2"
        example_inputs = get_example_inputs(model_name_or_path)
        benchmark(model_name_or_path, config, example_inputs=example_inputs)

    def test_int8_torchscript(self):
        config = BenchmarkConfig(
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=-1,
            torchscript=True,
        )
        model_name_or_path = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/distilbert_sst2_int8"):
            model_name_or_path = "/tf_dataset2/models/nlp_toolkit/distilbert_sst2_int8"
        example_inputs = get_example_inputs(model_name_or_path)
        benchmark(model_name_or_path, config, example_inputs=example_inputs)

    def test_generate(self):
        config = BenchmarkConfig(
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=1,
            torchscript=False,
            generate=True,
            max_length=195,
        )
        model_name = "adasnew/t5-small-xsum"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/t5-small-xsum"):
            model_name = "/tf_dataset2/models/nlp_toolkit/t5-small-xsum"
        example_inputs = get_example_inputs(model_name, dataset_name='lambada')
        example_inputs = example_inputs['input_ids'][0].to('cpu').unsqueeze(0) # for UT coverage
        benchmark(model_name, config, example_inputs=example_inputs)

    def test_int8_generate(self):
        config = BenchmarkConfig(
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=1,
            torchscript=False,
            generate=True,
            max_length=195,
        )
        model_name = "Intel/t5-small-xsum-int8-dynamic"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/t5-small-xsum-int8-dynamic"):
            model_name = "/tf_dataset2/models/nlp_toolkit/t5-small-xsum-int8-dynamic"
        example_inputs = get_example_inputs(model_name, dataset_name='lambada')
        benchmark(model_name, config, example_inputs=example_inputs)

    def test_torchscript_generate(self):
        # test_int8_torchscript_generate will fail when torch.jit.trace, deleted.
        config = BenchmarkConfig(
            batch_size=16,
            cores_per_instance=4,
            num_of_instance=1,
            torchscript=True,
            generate=True,
            max_length=195,
        )
        model_name = "adasnew/t5-small-xsum"
        if os.path.exists("/tf_dataset2/models/nlp_toolkit/t5-small-xsum"):
            model_name = "/tf_dataset2/models/nlp_toolkit/t5-small-xsum"
        example_inputs = get_example_inputs(model_name, dataset_name='lambada')
        benchmark(model_name, config, example_inputs=example_inputs)

if __name__ == "__main__":
    unittest.main()
