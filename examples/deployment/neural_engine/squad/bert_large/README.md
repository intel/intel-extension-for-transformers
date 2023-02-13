Step-by-Step
========
This document describes the end-to-end workflow for Huggingface model [BERT Large](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) with Neural Engine backend.

# Prerequisite

## 1.Environment​
### Prepare Python Environment
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


### Environment Variables 
Preload libjemalloc.so can improve the performance when multi instance.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instance.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
## 2.Prepare Dataset

```shell
python prepare_dataset.py --dataset_name=squad --output_dir=./data
```

## 3.Prepare Model
Neural Engine can parse ONNX model and IR.  
You could get fp32 ONNX model by setting precision=fp32, command is as follows:
```shell
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=fp32
```
By setting precision=int8 you could get PTQ int8 model and setting precision=bf16 to get bf16 model.
```shell
bash prepare_model.sh --input_model=bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name=squad --task_name=squad --output_dir=./model_and_tokenizer --precision=int8
```
You could also compile the model to IR using python API as follows:
```
from intel_extension_for_transformers.backends.neural_engine.compile import compile
graph = compile('./model_and_tokenizer/int8-model.onnx')
graph.save('./ir')
```

# Benchmark

## 1.Accuracy
Python API Command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1
```
Shell script is also avaiable:
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1
```

If you just want a quick start, you could try a small set of dataset, like this:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
```
or run shell
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=1 --max_eval_samples=10
```
> **Note**: The accuracy of partial dataset is unauthentic.

## 2.Performance
Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=1 --seq_len=384
```
Shell script is also avaiable:
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=1 --seq_len=384
```
You could use C++ API as well. First, you need to compile the model to IR. And then, you could run C++. 
> **Note**: The warmup below is recommended to be 1/10 of iterations and no less than 3.
```
export GLOG_minloglevel=2
export OMP_NUM_THREADS=<cpu_cores>
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export UNIFIED_BUFFER=1
numactl -C 0-<cpu_cores-1> <intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/bin/neural_engine
--batch_size=<batch_size> --iterations=<iterations> --w=<warmup>
--seq_len=384 --config=./ir/conf.yaml --weight=./ir/model.bin
```
