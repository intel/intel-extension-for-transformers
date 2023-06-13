# Step-by-Step

The implementation is based on [Length Adaptive Transformer](https://github.com/clovaai/length-adaptive-transformer)'s work.
Currently, it supports BERT based transformers.

[QuaLA-MiniLM: A Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114) has been accepted by NeurIPS 2022. Our quantized length-adaptive MiniLM model (QuaLA-MiniLM) is trained only once, dynamically fits any inference scenario, and achieves an accuracy-efficiency trade-off superior to any other efficient approaches per any computational budget on the SQuAD1.1 dataset (up to x8.8 speedup with <1% accuracy loss). The following shows how to reproduce this work and we also provide the [notebook tutorials](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/tutorials/pytorch/question-answering/Dynamic_MiniLM_SQuAD.ipynb).


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
cd <intel_extension_for_transformers_folder>/examples/huggingface/pytorch/question-answering/deployment/squad/length_adaptive_transformer
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
bash run_bert_large.sh --model=sguskin/dynamic-minilmv2-L6-H384-squad1.1 --dataset=squad --precision=fp32 --mode=throughput
```

By setting `precision=int8` you could get PTQ int8 model
```shell
bash run_bert_large.sh --model=sguskin/dynamic-minilmv2-L6-H384-squad1.1 --dataset=mrpc --precision=int8 --mode=throughput
```

Python API is also available to generate onnx model:

For FP32:
```shell
python run_qa.py --model_name_or_path "sguskin/dynamic-minilmv2-L6-H384-squad1.1" --dataset_name squad --do_train --do_eval --output_dir model_and_tokenizer --overwrite_output_dir --length_config "(269, 253, 252, 202, 104, 34)" --overwrite_cache --to_onnx
```

For INT8:
```shell
python run_qa.py --model_name_or_path "sguskin/dynamic-minilmv2-L6-H384-squad1.1" --dataset_name squad --do_train --do_eval --output_dir model_and_tokenizer --overwrite_output_dir --length_config "(269, 253, 252, 202, 104, 34)" --overwrite_cache --to_onnx --tune --quantization_approach PostTrainingStatic
```

For BF16:
```shell
python run_qa.py --model_name_or_path "sguskin/dynamic-minilmv2-L6-H384-squad1.1" --dataset_name squad --do_train --do_eval --output_dir model_and_tokenizer --overwrite_output_dir --length_config "(269, 253, 252, 202, 104, 34)" --overwrite_cache --to_onnx --enable_bf16
```


You could also compile the model to IR using python API as follows:
```python
from intel_extension_for_transformers.backends.neural_engine.compile import compile
graph = compile('./model_and_tokenizer/int8-model.onnx')
graph.save('./ir')
```
# Benchmark
## 1.Accuracy

Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --dataset_name=squad --batch_size=8
```
If you just want a quick start, you could try a small set of dataset, like this:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=accuracy --dataset_name=squad --batch_size=1 --max_eval_samples=10
```
> **Note**: The accuracy of partial dataset is unauthentic.

## Performance

Python API command as follows:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=1 --seq_len=384
```
