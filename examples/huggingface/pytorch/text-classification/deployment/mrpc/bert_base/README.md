Step-by-Step
=======
This document describes the end-to-end workflow for Huggingface model [BERT Base Uncased](https://huggingface.co/textattack/bert-base-uncased-MRPC) with Neural Engine backend.

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
cd <intel_extension_for_transformers_folder>/examples/huggingface/pytorch/text-classification/deployment/mrpc/bert_base
pip install -r requirements.txt
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11

## Environment Variables (Optional)
```shell
# Preload libjemalloc.so may improve the performance when inference under multi instance.
conda install jemalloc==5.2.1 -c conda-forge -y
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so

# Using weight sharing can save memory and may improve the performance when multi instances.
export WEIGHT_SHARING=1
export INST_NUM=<inst num>
```
>**Note**: This step is optional.

# Inference Pipeline

Neural Engine can parse ONNX model and Neural Engine IR. 
We provide with three `mode`s: `accuracy`, `throughput` or `latency`. For throughput mode, we will use multi-instance with 4cores/instance occupying one socket.
You can run fp32 model inference by setting `precision=fp32`, command as follows:
```shell
bash run_bert_base.sh --model=textattack/bert-base-uncased-MRPC  --dataset=mrpc --precision=fp32 --mode=throughput
```
By setting `precision=int8` you could get PTQ int8 model and setting `precision=bf16` to get bf16 model.
```shell
bash run_bert_base.sh --model=textattack/bert-base-uncased-MRPC  --dataset=mrpc --precision=int8 --mode=throughput
```
By setting `precision=dynamic_int8`, you could benchmark dynamic quantized int8 model.
```shell
bash run_bert_base.sh --model=textattack/bert-base-uncased-MRPC  --dataset=mrpc --precision=dynamic_int8 --mode=throughput
```


You could also compile the model to IR using python API as follows:
```python
from intel_extension_for_transformers.backends.neural_engine.compile import compile
graph = compile('./model_and_tokenizer/int8-model.onnx')
graph.save('./ir')
```

# Benchmark
If you want to run local onnx model inference, we provide with python API and C++ API. To use C++ API, you need to transfer to model ir fisrt.

By setting `--dynamic_quanzite` for FP32 model, you could benchmark dynamic quantize int8 model.
## Accuracy
Python API Command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --dataset_name=glue --task_name=mrpc --batch_size=8
```

If you just want a quick start, you could try a small set of dataset, like this:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --dataset_name=glue --task_name=mrpc --batch_size=8 --max_eval_samples=10
```

>**Note**: The accuracy of partial dataset is unauthentic.

## Performance
Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --dataset_name=glue --task_name=mrpc  --batch_size=1 --seq_len=128
```

You could use C++ API as well. First, you need to compile the model to IR. And then, you could run C++.

> **Note**: The warmup below is recommended to be 1/10 of iterations and no less than 3.
```shell
export GLOG_minloglevel=2
export OMP_NUM_THREADS=<cpu_cores>
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export UNIFIED_BUFFER=1
numactl -C 0-<cpu_cores-1> neural_engine \
  --batch_size=<batch_size> --iterations=<iterations> --w=<warmup> \
  --seq_len=128 --config=./ir/conf.yaml --weight=./ir/model.bin
```
