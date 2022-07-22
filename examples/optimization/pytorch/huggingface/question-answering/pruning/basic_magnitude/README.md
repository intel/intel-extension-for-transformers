# Question answering

The script `run_qa.py` provides the pruning approach (Magnitude) based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Here is how to run the script:

```
python run_qa.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --prune \
    --target_sparsity 0.1 \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output
```

We also supported Distributed Data Parallel training on single node and multi nodes settings for pruning. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be lanuched in each node, and all the commands should be the same except for *`<NUM_NODES>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    run_qa.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --prune \
    --target_sparsity 0.1 \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output
```

### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
