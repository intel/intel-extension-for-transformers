Step-by-Step​
============
Distillation script [run_qa.py](./run_qa.py) provides the distillation approach based on [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

# Prerequisite​

## 1. Create Environment​
Recommend python 3.7 or higher version.
```shell
pip install intel-extension-for-transformers
pip install -r requirements.txt
```

# Distillation
## 1. Single-Node Distillation

Here is how to run the script:
```
python run_qa.py \
    --model_name_or_path distilbert-base-uncased \
    --teacher_model_name_or_path csarron/bert-base-uncased-squad-v1 \
    --dataset_name squad \
    --distillation \
    --do_train \
    --do_eval \
    --no_cuda \
    --run_teacher_logits \
    --output_dir ./tmp/squad_output
```


## 2. Distributed Data Parallel Support

We also supported Distributed Data Parallel training on single node and multi nodes settings for distillation. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
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
    run_qa.py \
        --model_name_or_path distilbert-base-uncased \
        --teacher_model_name_or_path csarron/bert-base-uncased-squad-v1 \
        --dataset_name squad \
        --distillation \
        --do_train \
        --do_eval \
        --no_cuda \
        --run_teacher_logits \
        --output_dir ./tmp/squad_output
```
