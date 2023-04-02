Step-by-Step
=========
This document describes the end-to-end workflow for Huggingface model [VIT](https://huggingface.co/google/vit-base-patch16-224) with Neural Engine backend.

# Prerequisite

## Installation
### Install python environment
Create a new python environment
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
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/sst2/bert_mini
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11


### Environment Variables
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instances.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
## Inference Pipeline
Neural Engine can parse ONNX model and Neural Engine IR.
We provide with three mode: accuracy, throughput or latency. For throughput mode, we will use multi-instance with 4cores/instance occupying one socket.
You can run fp32 model inference by setting `precision=fp32`, command as follows:
```shell
bash run_vit.sh --input_model=google/vit-base-patch16-224  --task_name=imagenet-1k --precision=fp32
```
By setting `precision=int8` you could get PTQ int8 model and setting `precision=bf16` to get bf16 model.
```shell
bash run_vit.sh --input_model=google/vit-base-patch16-224  --task_name=imagenet-1k --precision=int8
```
Note: the input_model could be changed from a vit base model to a vit large model.

## Benchmark
If you want to run local onnx model inference, we provide with python API and C++ API. To use C++ API, you need to transfer to model ir fisrt.

By setting --dynamic_quanzite for FP32 model, you could benchmark dynamic quantize int8 model. 
### Accuracy  

Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --data_dir=./data --batch_size=8
  ```

### Performance
Python API command as follows:
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=8 --seq_len=128
  ```

  Or compile framwork model to IR using python API
  ```
  from intel_extension_for_transformers.backends.neural_engine.compile import compile
  graph = compile('./model_and_tokenizer/int8-model.onnx')
  graph.save('./ir')
  ```
