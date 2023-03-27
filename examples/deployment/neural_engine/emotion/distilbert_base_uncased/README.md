# Step-by-Step

# Prerequisite

## 1. Installation
1.1 Install python environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```
Check the gcc version using $gcc-v, make sure the gcc version is higher than 9.0.
If not, you need to update gcc by yourself.
Make sure you have the autoconf installed.
If not, you need to install autoconf by yourself.
Make sure the cmake version is 3 rather than 2.
If not, you need to install cmake.
```shell
cmake --version
conda install cmake
sudo apt install autoconf
```
Install Intel Extension for Transformers from Source Code
```shell
cd <intel_extension_for_transformers_folder>
git submodule update --init --recursive
python setup.py install
```
Install package for examples
```shell
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/emotion/distilbert_base_uncased
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11


## 2. Environment Variables
Preload libjemalloc.so can improve the performance when multi instance.
```
export LD_PRELOAD=<intel_extension_for_transformers_folder>/intel_extension_for_transformers/backends/neural_engine/executor/third_party/jemalloc/lib/libjemalloc.so
```
Using weight sharing can save memory and improve the performance when multi instance.
```
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
# Inference Pipeline

Neural Engine can parse ONNX model and Neural Engine IR. 
We provide with three mode: accuracy, throughput or latency. For throughput mode, we will use multi-instance with 4cores/instance occupying one socket.
You can run fp32 model inference by setting `precision=fp32`, command as follows:
```shell
GLOG_minloglevel=2 bash run_emotion.sh --model=bhadresh-savani/distilbert-base-uncased-emotion --dataset=emotion --precision=fp32 
```
By setting `precision=int8` you could get PTQ int8 model and setting `precision=bf16` to get bf16 model.
```shell
GLOG_minloglevel=2 bash run_emotion.sh -m bhadresh-savani/distilbert-base-uncased-emotion -d emotion -p int8
```

# Benchmark
If you want to run local onnx model inference, we provide with python API and C++ API. To use C++ API, you need to transfer to model ir fisrt.
## 1. Accuracy 
By setting --dynamic_quanzite for FP32 model, you could benchmark dynamic quantize int8 model.

```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizerint8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy--data_dir=./data --batch_size=4
```
or run shell
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data--batch_size=4
```

## 2. Performance 
By setting --dynamic_quanzite for FP32 model, you could benchmark dynamic quantize int8 model.

```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizerint8-model.onnx --mode=performance --batch_size=8 --seq_len=128
```
or run shell
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=8 --seq_len=128
```
or compile framwork model to IR using python API
```
from intel_extension_for_transformers.backends.neural_engine.compile importcompile
graph = compile('./model_and_tokenizer/int8-model.onnx')
graph.save('./ir')
```
and run C++  
The warmup below is recommended to be 1/10 of iterations and no less than 3.
```
export GLOG_minloglevel=2
export OMP_NUM_THREADS=<cpu_cores>
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export UNIFIED_BUFFER=1
numactl -C 0-<cpu_cores-1> <intel_extension_for_transformers_folder>intel_extension_for_transformers/backends/neural_engine/bin/neural_engine
--batch_size=<batch_size> --iterations=<iterations> --w=<warmup>
--seq_len=128 --config=./ir/conf.yaml --weight=./ir/model.bin
```
