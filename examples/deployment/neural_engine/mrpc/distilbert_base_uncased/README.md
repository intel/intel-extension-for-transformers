Step-by-Step
=======
This document describes the end-to-end workflow for Huggingface model [DistilBERT Base Uncased](https://huggingface.co/textattack/distilbert-base-uncased-MRPC) with Neural Engine backend.

# Prerequisite
## Prepare Python Environment
Create a new python environment
```shell
conda create -n <env name> python=3.8
conda activate <env name>
```

Check the gcc version using `gcc-v`, make sure the `gcc` version is higher than 9.0.
If not, you need to update `gcc` by yourself.
Make sure you have the `autoconf` installed.
If not, you need to install `autoconf` by yourself.
Make sure the `cmake` version is 3 rather than 2.
If not, you need to install `cmake`.
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
cd <intel_extension_for_transformers_folder>/examples/deployment/neural_engine/mrpc/bert_base
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11

## Environment Variables
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
bash run_bert_mini.sh --model=textattack/distilbert-base-uncased-MRPC --dataset=mrpc --precision=fp32
```

By setting `precision=int8` you could get PTQ int8 model and setting `precision=bf16` to get bf16 model.
```shell
bash run_bert_mini.sh --model=textattack/distilbert-base-uncased-MRPC --dataset=mrpc --precision=int8
```

You could also compile the model to IR using python API as follows:
```python
from intel_extension_for_transformers.backends.neural_engine.compile import compile
graph = compile('./model_and_tokenizer/int8-model.onnx')
graph.save('./ir')
```

# Benchmark
<<<<<<< HEAD
If you want to run local onnx model inference, we provide with python API and C++ API. To use C++ API, you need to transfer to model ir fisrt.

By setting --dynamic_quanzite for FP32 model, you could benchmark dynamic quantize int8 model.
## Accuracy
=======
Throught setting --dynamic_quanzite for FP32 model, you could benchmark dynamic quantize int8 model.
## 1. Accuracy
>>>>>>> develop
Python API Command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8
```

Shell script is also avaiable:
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8
```

If you just want a quick start, you could try a small set of dataset, like this:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8 --max_eval_samples=10
```

or run shell:
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8 --max_eval_samples=10
```

>**Note**: The accuracy of partial dataset is unauthentic.

## Performance
Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=8 --seq_len=128
```

Shell script is also avaiable:
```shell
bash run_benchmark.sh --input_model=./model_and_tokenizer/int8-model.onnx  --mode=performance --batch_size=8 --seq_len=128
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
--seq_len=128 --config=./ir/conf.yaml --weight=./ir/model.bin
```
