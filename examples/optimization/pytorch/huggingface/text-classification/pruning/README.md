Step-by-step
============

This example shows basic magnitude pruning. Basic magnitude pruning is the simplest weight pruning algorithm. After each training, the link with the smallest weight is removed. Thus the salience of a link is just the absolute size of its weight.

# Prerequisite​

## 1. Environment
```
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Run

## Step 1: Pruning

The script `run_glue.py` provides the magnitude pruning approach based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

```
python run_glue.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --prune \
    --do_train \
    --do_eval \
    --output_dir ./tmp/sst2_output \
    --overwrite_output_dir
```

## Step 2: Distributed Data Parallel Training

We supporte Distributed Data Parallel training on single node and multi nodes settings for pruning. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
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

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    run_glue.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --prune \
    --do_train \
    --do_eval \
    --output_dir ./tmp/sst2_output \
    --overwrite_output_dir
```
