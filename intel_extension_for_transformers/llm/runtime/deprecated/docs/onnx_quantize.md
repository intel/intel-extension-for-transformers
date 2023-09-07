# Quantize a ONNX model to engine low precision/int8 IR

## Design
Quantizing a ONNX model to engine low precision/int8 IR has two steps: 1. Convert ONNX model to engine float IR; 2. Quantize float IR to low precision/int8 IR. The first step will be finished in engine compile. We focus on the second step how to quantize a float engine IR to low precision IR in INC.

## Prerequisite
### Install environment
```shell
cd ${HOME}/examples/huggingface/pytorch/text-classification/deployment/mrpc/distilbert_base_uncased
conda create -n <env name> python=3.7
conda activate <env name>
pip install -r requirements.txt
```
**Note**: Recommend install protobuf <= 3.20.0 if use onnxruntime <= 1.11


### Prepare Dataset
```python
python prepare_dataset.py --tasks='MRPC' --output_dir=./data
```
### Prepare ONNX model
```shell
bash prepare_model.sh
```

## Run tuning and benchmark
Users can run shell to tune model by optimization module and get its accuracy and output onnx model.
### 1. To get the tuned model and its accuracy:
```shell
bash prepare_model.sh --input_model=moshew/bert-mini-sst2-distilled  --task_name=sst2 --output_dir=./model_and_tokenizer --precision=int8
```

### 2. To get the benchmark of tuned model:
```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx  --tokenizer_dir=./model_and_tokenizer --mode=accuracy --data_dir=./data --batch_size=8
```

```shell
GLOG_minloglevel=2 python run_executor.py --input_model=./model_and_tokenizer/int8-model.onnx --mode=performance --batch_size=8 --seq_len=128
```
