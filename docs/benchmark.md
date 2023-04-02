Benchmark
======
1. [Introduction](#introduction)
2. [Get Started](#get-started-with-benchmark-api)
3. [Examples](#examples)

    3.1. [Stock Pytorch Model](#stock-pytorch-model)

    3.2. [IPEX Model](#ipex-model)

    3.3. [Benchmark Output](#benchmark-output)

## Introduction

The Benchmark is used to measure the model performance with the objective settings. It is inherited from Intel® Neural Compressor [Benchmark](https://github.com/intel/neural-compressor/blob/master/docs/source/benchmark.md).

## Get Started with Benchmark API
The class `BenchmarkConfig` allows users to adjust the following parameters with objective settings to measure model performance:

`backend` (str, optional): the backend used for benchmark. Defaults to "torch". \
`warmup` (int, optional): number of iters to skip when collecting latency. Defaults to 5. \
`iteration` (int, optional): total iters when collecting latency. Defaults to 20. \
`cores_per_instance` (int, optional): the core number for 1 instance. Defaults to 4. \
`num_of_instance` (int, optional): the instance number. Defaults to -1. \
`torchscript` (bool, optional): enable it if you want to jit trace it before benchmarking. Defaults to False. \
`generate` (bool, optional): enable it if you want to use model generate when benchmarking. Defaults to False.

>**Note**: Benchmark provides capability to automatically run with multiple instance through `cores_per_instance` and `num_of_instance` config (CPU only). Please make sure `cores_per_instance * num_of_instance` must be less than CPU physical core numbers. 

## Examples
Example inputs or a dataloader is required for benchmark.
```py
def get_example_inputs(model_name, dataset_name='sst2'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name, split='validation')
    text = dataset[0]['text'] if dataset_name=='lambada' else dataset[0]['sentence']
    example_inputs = tokenizer(text, padding='max_length', max_length=195, return_tensors='pt')
    return example_inputs
```
### Stock Pytorch Model
```py
from intel_extension_for_transformers.optimization import BenchmarkConfig
from intel_extension_for_transformers.optimization.benchmark import benchmark

config = BenchmarkConfig(
    batch_size=16,
    cores_per_instance=4,
    num_of_instance=-1,
)
example_inputs = get_example_inputs(model_name_or_path)
benchmark(model_name_or_path, config, example_inputs=example_inputs)
```
### IPEX Model
```py
from intel_extension_for_transformers.optimization import BenchmarkConfig
from intel_extension_for_transformers.optimization.benchmark import benchmark

config = BenchmarkConfig(
    backend='ipex',
    batch_size=16,
    cores_per_instance=4,
    num_of_instance=-1,
)
example_inputs = get_example_inputs(model_name_or_path)
benchmark(model_name_or_path, config, example_inputs=example_inputs)
```
### Benchmark Output
```bash
**********************************************
|****Multiple Instance Benchmark Summary*****|
+---------------------------------+----------+
|              Items              |  Result  |
+---------------------------------+----------+
| Latency average [second/sample] |  0.003   |
| Throughput sum [samples/second] | 5071.933 |
+---------------------------------+----------+
```
