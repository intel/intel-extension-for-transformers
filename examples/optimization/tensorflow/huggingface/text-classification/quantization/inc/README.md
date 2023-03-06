Step-by-Step
=========

This document describes the step-by-step instructions for reproducing the quantization on models for the text classification (GLUE) tasks.

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

# Prerequisite
## 1. Installation

Make sure you have installed Intel® Extension for Transformers and all the dependencies in the current example:

```shell
pip install intel-extension-for-transformers
cd ptq
pip install -r requirements.txt
```

# Run

Here are two options: running with the shell script or running with the python script. Basically, they are equivalent and the shell script just wraps the invocation of the python script and is more concise and easy for users to get started.

## 1. Run Command (Shell)

- Topology:
   - bert_base_mrpc_static
   - xlnet_mrpc
   - albert_large_mrpc
   - legalbert_mrpc

- To get the int8 model

   ```
   cd ptq
   bash run_tuning.sh  --topology=[topology] --output_model=./saved_int8
   ```

- To benchmark the int8 model

   ```
   cd ptq
   bash run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
   ```

## 2. Run Command (Python)

- model_name_or_path: 
   - bert-base-cased-finetuned-mrpc
   - xlnet-base-cased
   - albert-large-v2
   - nlpaueb/legal-bert-small-uncased

- To get int8 model

```
python run_glue.py     
    --model_name_or_path [model_name_or_path] \
    --task_name mrpc \     
    --tune \     
    --quantization_approach PostTrainingStatic \     
    --do_train \     
    --do_eval \     
    --output_dir ./saved_result \  
    --overwrite_output_dir
```
 - To reload int8 model

```
python run_glue.py     
    --model_name_or_path [model_name_or_path] \
    --task_name mrpc \     
    --benchmark \
    --int8 \
    --do_eval \     
    --output_dir ./saved_result \  
    --overwrite_output_dir
```

> **Notes**:
 - quantization_approach in Tensorflow consist of `PostTrainingStatic`, `QuantizationAwareTraining`.
 - task_name consist of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.


# Multi-node Usage

We also supported Distributed Data Parallel training on multi nodes settings for quantization.

> **Note**: multi node settings boost performance in the training process and may not show good performance with PostTrainingStatic quantization strategy

The default strategy we used is `MultiWorkerMirroredStrategy` in Tensorflow, and with `task_type` set as "worker", we are expected to pass following extra parameters to the script:

* `worker`: a string of your worker ip addresses which is separated by comma and there should not be space between each two of them

* `task_index`: 0 should be set on the chief node (leader) and 1, 2, 3... should be set as the rank of other follower nodes

## Multi-node Example

### 1. Get Int8 Model

* On leader node

```
bash run_tuning.sh --topology=bert_base_mrpc_static --output_model=./saved_int8 --worker="localhost:12345,localhost:23456"  --task_index=0
```

* On follower node

```
bash run_tuning.sh --topology=bert_base_mrpc_static --output_model=./saved_int8 --worker="localhost:12345,localhost:23456"  --task_index=1
```

Please replace the worker ip address list with your own.

### 2. Reload Int8 Model

* On leader node

```
bash run_benchmark.sh --topology=bert_base_mrpc_static --config=./saved_int8 --mode=benchmark --int8=true --worker="localhost:12345,localhost:23456"  --task_index=0
```

* On follower node

```
bash run_benchmark.sh --topology=bert_base_mrpc_static --config=./saved_int8 --mode=benchmark --int8=true --worker="localhost:12345,localhost:23456"  --task_index=1
```

Please replace the worker ip address list with your own.




