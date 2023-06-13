Step-by-Step
========
This document describes the end-to-end workflow for Huggingface model [Distilbert Base](https://huggingface.co/distilbert-base-uncased-distilled-squad) with IPEX backend.
# Prerequisite
## Prepare Python Environment
Create a python environment, optionally with autoconf for jemalloc support.
```shell
conda create -n <env name> python=3.8 [autoconf]
conda activate <env name>
```

Check that `gcc` version is higher than 9.0.
```shell
gcc -v
```

Install Intel® Extension for Transformers, please refer to [installation](/docs/installation.md).
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Or, install from source code
cd <intel_extension_for_transformers_folder>
pip install -v .
```

Install required dependencies for this example
```shell
cd <intel_extension_for_transformers_folder>/examples/examples/huggingface/pytorch/question-answering/deployment/squad/ipex/distillbert_base_uncased
pip install -r requirements.txt
```

# Inference Pipeline
Neural Engine can parse ONNX model and Neural Engine IR. 
We provide with three `mode`s: `accuracy`, `throughput` or `latency`. For throughput mode, we will use multi-instance with 4cores/instance occupying one socket.
You can run fp32 model inference by setting `precision=fp32`, command as follows:

```shell
bash run_distilbert.sh --model=distilbert-base-uncased-distilled-squad --dataset=squad --precision=fp32 --mode=throughput
```

By setting `precision=int8` you could get PTQ int8 model and setting `precision=bf16` to get bf16 model.
```shell
bash run_distilbert.sh --model=distilbert-base-uncased-distilled-squad --dataset=squad --precision=int8 --mode=throughput
```
# Benchmark
## Accuracy
Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1
  ```
  If you just want a quick start, you can run only a part of dataset.
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
  ```
  But the accuracy of quick start is unauthentic.

## Performance
Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=performance --batch_size=1 --seq_len=384
  ```
