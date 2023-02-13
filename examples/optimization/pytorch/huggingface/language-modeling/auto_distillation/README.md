Step-by-Step​
============

Auto Distillation training script [`run_mlm_autodistillation.py`](./run_mlm_autodistillation.py) is based on [`run_mlm.py`](https://github.com/IntelLabs/Model-Compression-Research-Package/blob/main/examples/transformers/language-modeling/run_mlm.py) of Model-Compression-Research-Package by IntelLabs.

# Prerequisite​

## 1. Environment​
Recommend python 3.7 or higher version.
```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset
Datasets are downloaded and processed using `🤗 Datasets` package.

# Auto Distillation
## 1. Usage
The script `run_mlm_autodistillation.py` can be used for auto distillation of `🤗 Transformers` models.

### 1.1 MobileBERT

Search best model architectures based on [google/mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) on English Wikipedia and BookCorpus datasets with [bert-base-uncased](https://huggingface.co/bert-base-uncased) as the teacher model using the following command:

``` bash
python run_mlm_autodistillation.py \
    --config_name mobilebert_config.json \
    --tokenizer_name google/mobilebert-uncased \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --auto_distillation \
    --teacher_config_name bertlarge_config.json \
    --teacher_model_name_or_path bert-large-uncased \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```

### 1.2 BERT-Tiny

Search best model architectures based on [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) on English Wikipedia and BookCorpus datasets with [bert-base-uncased](https://huggingface.co/bert-base-uncased) as the teacher model using the following command:

``` bash
python run_mlm_autodistillation.py \
    --config_name bert-tiny_config.json \
    --tokenizer_name prajjwal1/bert-tiny \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --auto_distillation \
    --teacher_config_name bert-base-uncased_config.json \
    --teacher_model_name_or_path bert-base-uncased \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```

## 2. Distributed Data Parallel Support

We also supported Distributed Data Parallel training on single node and multi nodes settings for Auto Distillation. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, to search best model architectures based on [google/mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) through Distributed Data Parallel training, bash command will look like the following, where
<br>
*`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case,
<br>
*`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1,
<br>
*`<NUM_NODES>`* is the number of nodes to use,
<br>
*`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
> Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

``` bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    run_mlm_autodistillation.py \
    --config_name mobilebert_config.json \
    --tokenizer_name google/mobilebert-uncased \
    --datasets_name_config wikipedia:20200501.en bookcorpusopen \
    --do_train --do_eval --auto_distillation \
    --teacher_config_name bertlarge_config.json \
    --teacher_model_name_or_path bert-large-uncased \
    --data_process_type segment_pair_nsp \
    --max_seq_length 128 \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```
