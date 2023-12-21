# Sparse model Step-by-Step
Here is an example of blocked sparsity and quantization of Bert Mini, sparse ratio is 90%.
Intel® Extension for Transformers provided a high-performance sparse matrix multiplication library – Transformers-accelerated Libraries and demonstrated the performance improvement of sparse outweigh the accuracy loss.
This sparse solution is a software-based solution and utilizes the Intel instructions. More sparse examples will be released in the future.
# Prerequisite

## Installation
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
pip install -r requirements.txt
pip install -v .
```

Install required dependencies for this example
```shell
cd <intel_extension_for_transformers_folder>/examples/huggingface/pytorch/text-classification/deployment/sparse/bert_mini
pip install -r requirements.txt
pip install transformers==4.34.1
```
>**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11

>**Note**: Please use transformers no higher than 4.34.1


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
bash run_bert_mini.sh --model=Intel/bert-mini-sst2-distilled-sparse-90-1X4-block --dataset=sst2 --precision=fp32 --mode=throughput
```
By setting `precision=int8` you could get PTQ int8 model and setting `precision=bf16` to get bf16 model.
```shell
bash run_bert_mini.sh --model=Intel/bert-mini-sst2-distilled-sparse-90-1X4-block --dataset=sst2 --precision=int8 --mode=throughput
```

By setting `precision=dynamic_int8`, you could benchmark dynamic quantized int8 model.
```shell
bash run_bert_mini.sh --model=Intel/bert-mini-sst2-distilled-sparse-90-1X4-block --dataset=sst2 --precision=dynamic_int8 --mode=throughput
```

### Benchmark
Neural Engine will automatically detect weight structured sparse ratio, as long as it beyond 70% (since normally get performance gain when sparse ratio beyond 70%). Neural Engine will call [Transformers-accelerated Libraries](/intel_extension_for_transformers/llm/runtime/deprecated/kernels) and high performance layernorm op with transpose mode to improve inference performance.
Before using Python API to benchmark, need to transpose onnx model to IR, command as follows:
```shell
python export_transpose_ir.py --input_model=./model_and_tokenizer/int8-model.onnx --output_dir=./sparse_int8_ir
```

## Accuracy
  run python
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./sparse_int8_ir  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --dataset_name=glue --task_name=sst2 --batch_size=8
  ```

## Performance
  run python
  
  ```shell
  GLOG_minloglevel=2 python run_executor.py --input_model=./sparse_int8_ir --mode=performance --batch_size=1 --seq_len=128 --dataset_name=glue --task_name=sst2
  ```
  
  or run C++
  The warmup below is recommended to be 1/10 of iterations and no less than 3.
  
  ```shell
  export GLOG_minloglevel=2
  export OMP_NUM_THREADS=<cpu_cores>
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  export UNIFIED_BUFFER=1
  numactl -C 0-<cpu_cores-1> neural_engine \
    --batch_size=<batch_size> --iterations=<iterations> --w=<warmup> \
    --seq_len=128 --config=./sparse_int8_ir/conf.yaml --weight=./sparse_int8_ir/model.bin
  ```
