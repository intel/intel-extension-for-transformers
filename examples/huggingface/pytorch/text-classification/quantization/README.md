Step-by-Step​
============
The script `run_glue.py` provides three quantization approaches (PostTrainingStatic, PostTrainingStatic and QuantizationAwareTraining) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version.
```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run
## 1. Quantization
GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:
 - To get int8 model

```
python run_glue.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --tune \
    --quantization_approach PostTrainingStatic \
    --do_train \
    --do_eval \
    --output_dir ./saved_result \
    --overwrite_output_dir
```
 - To reload int8 model

```
python run_glue.py \
    --model_name_or_path ./saved_result \
    --task_name sst2 \
    --benchmark \
    --int8 \
    --do_eval \
    --output_dir ./tmp/sst2_output \
    --overwrite_output_dir
```

**Notes**: 
 - Choice of `quantization_approach` can be `PostTrainingDynamic`, `PostTrainingStatic`, and `QuantizationAwareTraining`.
 - Choice of `task_name` can be `cola`, `sst2`, `mrpc`, `stsb`, `qqp`, `mnli`, `qnli`, `rte`, and `wnli`.

## 2. Distributed Data Parallel Support

We also supported Distributed Data Parallel training on single node and multi nodes settings for QAT. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
*`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case,
<br>
*`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1,
<br>
*`<NUM_NODES>`* is the number of nodes to use,
<br>
*`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>

 >**Note**: To use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.
```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    run_glue.py \
        --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
        --task_name sst2 \
        --tune \
        --quantization_approach QuantizationAwareTraining \
        --do_train \
        --do_eval \
        --output_dir ./saved_result \
        --overwrite_output_dir
```

## 3. Validated model list

|Task|Pretrained model|PostTrainingDynamic | PostTrainingStatic | QuantizationAwareTraining
|---|------------------------------------|---|---|---
|MRPC|textattack/bert-base-uncased-MRPC| ✅| ✅| ✅
|MRPC|textattack/albert-base-v2-MRPC| ✅| ✅| N/A
|SST-2|echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid| ✅| ✅| N/A
|SST-2|distilbert-base-uncased-finetuned-sst-2-english| ✅| ✅| N/A


## 3. Bash Command
### PostTrainingQuantization

|Category|Dataset|Topology
|---|---|------------------------------------
|BERT Base|MRPC|bert_base_mrpc_dynamic, bert_base_mrpc_static
|BERT Base|SST2|bert_base_SST-2_dynamic, bert_base_SST-2_static
|DISTILLBERT|SST2|distillbert_base_SST-2_dynamic, distillbert_base_SST-2_static
|ALBERT Base|SST2|albert_base_MRPC_dynamic, albert_base_MRPC_stati

 - To get int8 model

    ```
    cd ptq
    bash run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    cd ptq
    bash run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```

### QuantizationAwareTraining

- Topology: 
    - BERT-MRPC: bert_base_mrpc

 - To get int8 model

    ```
    cd qat
    bash run_tuning.sh  --topology=[topology] --output_model=./saved_int8
    ```

 - To reload int8 model

    ```
    cd qat
    bash run_benchmark.sh --topology=[topology] --config=./saved_int8 --mode=benchmark --int8=true
    ```
