Step-by-Step
========
This document describes the end-to-end workflow for Huggingface model [BERT Large](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) with IPEX backend.

# Prerequisite

## 1. Installation
### 1.1 Prepare Python Environment
Create a python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Make sure you have the autoconf installed. 
Also, `gcc` higher than 9.0, `cmake` higher than 3 is required.

```shell
gcc -v
cmake --version
conda install cmake
sudo apt install autoconf
```
Install Intel® Extension for Transformers, please refer to [installation](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md)
```shell
# Install from pypi
pip install intel-extension-for-transformers

# Install from source code
cd <intel_extension_for_transformers_folder>
git submodule update --init --recursive
python setup.py install
```
Install required dependencies for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/squad/bert_large
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11

### 1.2 Environment Variables
Preload libjemalloc.so can improve the performance when multi instance.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instance.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
## 2. Prepare Dataset and Pretrained Model

### 2.1 Get Dataset

```shell
python prepare_dataset.py --dataset_name=squad --output_dir=./data
```

### 2.2 Get Model
You can get a fp32 pytorch model from the optimization module by setting precision=fp32, command is as follows:
```shell
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=fp32
```

By setting precision=int8, you can get PTQ int8 JIT model.
```
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=int8
```

## Benchmark
### 2.1 Accuracy  
Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1
```
If you just want a quick start, you can run only a part of dataset.
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
```
But the accuracy of quick start is unauthentic.

### 2.2 Performance
Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer --mode=performance --batch_size=1 --seq_len=384
```
