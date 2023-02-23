# Question answering

This document is used to list steps of introducing [Prune Once For All](https://arxiv.org/abs/2111.05754) examples.

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the flag --version_2_with_negative.

The following example fine-tunes the pre-trained [90% sparse DistillBERT](Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa) on the SQuAD1.0 dataset by applying quantization aware-training, pattern lock pruning and distillation simultaneously.
 >**Note**: If you want to fine-tune the pre-trained model, you could change [conf_list](./run_qa.py#L832), remove the quantization_conf.
 
```
python run_qa.py \
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --teacher_model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --orchestrate_optimizations \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output \
    --overwrite_output_dir
```

We also supported Distributed Data Parallel training on single node and multi nodes settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    run_qa.py \
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --teacher_model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --orchestrate_optimizations \
    --do_train \
    --do_eval \
    --output_dir ./tmp/squad_output \
    --overwrite_output_dir
```

### Command

```
bash run_tuning.sh  --topology=topology
```

```
bash run_benchmark.sh --topology=topology --mode=benchmark
```
