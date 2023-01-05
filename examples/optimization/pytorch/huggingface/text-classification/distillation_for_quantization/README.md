# Text classification

## GLUE task

This document is used to illustrate how to run the distillation for quantization examples.
<br>
These examples will take a NLP model fine tuned on the down stream task, use its copy as a teacher model, and do distillation during the process of quantization aware training.
<br>
For more informations of this algorithm, please refer to the paper [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861)

# Prerequisite

## Python Version

Recommend python 3.7 or higher version.


## Install dependency

```shell
pip install -r requirements.txt
```

# Start running neural_compressor implementation of distillation for quantization

Below are example NLP tasks of distillation for quantization to quantize the fine tuned BERT model on the specific tasks.
<br>
It requires the pre-trained task specific model such as `yoshitomo-matsubara/bert-base-uncased-sst2` from yoshitomo-matsubara Huggingface portal as the teacher model for distillation, as well as for quantization.
<br>
The distillation configuration is specified in the [run_glue.py](./run_glue.py#L549), the quantization aware training configuration is also specified in [run_glue.py](./run_glue.py#L561).

## SST-2 task

```bash
python run_glue.py --task_name sst2 --model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --do_train --do_eval --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

## MNLI task

```bash
python run_glue.py --task_name mnli --model_name_or_path yoshitomo-matsubara/bert-base-uncased-mnli --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --do_train --do_eval --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

## QQP task

```bash
python run_glue.py --task_name qqp --model_name_or_path yoshitomo-matsubara/bert-base-uncased-qqp --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --do_train --do_eval --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

## QNLI task

```bash
python run_glue.py --task_name qnli --model_name_or_path yoshitomo-matsubara/bert-base-uncased-qnli --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --do_train --do_eval --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

# Distributed Data Parallel Support

We also supported Distributed Data Parallel training on single node and multi nodes settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    run_glue.py --task_name sst2 --model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
    --do_train --do_eval --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```