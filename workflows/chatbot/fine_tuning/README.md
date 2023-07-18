NeuralChat Fine-tuning
============

This example demonstrates how to finetune the pretrained large language model (LLM) with the instruction-following dataset for creating the NeuralChat, a chatbot that can conduct the textual conversation. Giving NeuralChat the textual instruction, it will respond with the textual response. This example have been validated on the 4th Gen Intel® Xeon® Processors, Sapphire Rapids.

# Prerequisite​

## 1. Environment​
Recommend python 3.9 or higher version.
```shell
pip install -r requirements.txt
# To use ccl as the distributed backend in distributed training on CPU requires to install below requirement.
python -m pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu
```

## 2. Prepare the Model

### LLaMA
To acquire the checkpoints and tokenizer, the user has two options: completing the [Google form](https://forms.gle/jk851eBVbX1m5TAv5) or attempting [the released model on Huggingface](https://huggingface.co/decapoda-research/llama-7b-hf). 

It should be noticed that the early version of LLama model's name in Transformers has resulted in many loading issues, please refer to this [revision history](https://github.com/huggingface/transformers/pull/21955). Therefore, Transformers has reorganized the code and rename LLaMA model as `Llama` in the model file. But the release model on Huggingface did not make modifications in react to this change. To avoid unexpexted confliction issues, we advise the user to modify the local `config.json` and `tokenizer_config.json` files according to the following recommendations:
1. The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`;
2. The `architectures` in `config.json` should be changed from `LLaMAForCausalLM` to `LlamaForCausalLM`.

### FLAN-T5
The user can obtain the [release model](https://huggingface.co/google/flan-t5-xl) from Huggingface.

## 3. Prepare Dataset
The instruction-following dataset is needed for the finetuning. We select two kinds of Datasets to conduct the finetuning process: general domain dataset and domain specific dataset.

1. General domain dataset: We use the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). In Alpaca, researchers have manually crafted 175 seed tasks to guide `text-davinci-003` in generating 52K instruction data for diverse tasks.

2. Domain-specific dataset: Inspired by Alpaca, we constructed a domain-specific dataset focusing on Business and Intel-related issues. We made minor modifications to the [prompt template](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) to proactively guide Alpaca in generating more Intel and Business related instruction data. The generated data could be find in `intel_domain.json`.

# Finetune

We employ the [LoRA approach](https://arxiv.org/pdf/2106.09685.pdf) to finetune the LLM efficiently, currently, FLAN-T5 and LLaMA are supported for finetuning.

## 1. Single Node Fine-tuning

For FLAN-T5, use the below command line for finetuning on the Alpaca dataset.

```bash
python finetune_seq2seq.py \
        --model_name_or_path "google/flan-t5-xl" \
        --bf16 True \
        --train_file "stanford_alpaca/alpaca_data.json" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1.0e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 5 \
        --logging_steps 10 \
        --save_steps 2000 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --output_dir ./flan-t5-xl_peft_finetuned_model \
        --peft lora
```

For LLaMA, use the below command line for finetuning on the Alpaca dataset.

```bash
python finetune_clm.py \
        --model_name_or_path "decapoda-research/llama-7b-hf" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./llama_peft_finetuned_model \
        --peft lora \
        --use_fast_tokenizer false \
```

For [MPT](https://huggingface.co/mosaicml/mpt-7b), use the below command line for finetuning on the Alpaca dataset. Only LORA supports MPT in PEFT perspective.it uses gpt-neox-20b tokenizer, so you need to define it in command line explicitly.This model also requires that trust_remote_code=True be passed to the from_pretrained method. This is because we use a custom MPT model architecture that is not yet part of the Hugging Face transformers package.

```bash
python finetune_clm.py \
        --model_name_or_path "mosaicml/mpt-7b" \
        --bf16 True \
        --train_file "/path/to/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --log_level info \
        --save_strategy epoch \
        --output_dir ./mpt_peft_finetuned_model \
        --peft lora \
        --trust_remote_code True \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
```

Where the `--dataset_concatenation` argument is a way to vastly accelerate the fine-tuning process through training samples concatenation. With several tokenized sentences concatenated into a longer and concentrated sentence as the training sample instead of having several training samples with different lengths, this way is more efficient due to the parallelism characteristic provided by the more concentrated training samples.

For finetuning on SPR, add `--bf16` argument will speedup the finetuning process without the loss of model's performance.
You could also indicate `--peft` to switch peft method in P-tuning, Prefix tuning, Prompt tuning, LLama Adapter, LoRA,
see https://github.com/huggingface/peft. Note for FLAN-T5, only LoRA is supported.

Add option **"--use_fast_tokenizer False"** when using latest transformers if you met failure in llama fast tokenizer for llama, The `tokenizer_class` in `tokenizer_config.json` should be changed from `LLaMATokenizer` to `LlamaTokenizer`

## 2. Multi-node Fine-tuning

We also supported Distributed Data Parallel finetuning on single node and multi-node settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, to finetune FLAN-T5 through Distributed Data Parallel training, bash command will look like the following, where
<br>
*`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case,
<br>
*`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1,
<br>
*`<NUM_NODES>`* is the number of nodes to use,
<br>
*`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
> Also please note that to use CPU for training in each node with multi-node settings, argument `--no_cuda` is mandatory, and `--xpu_backend ccl` is required if to use ccl as the distributed backend. In multi-node setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

``` bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
    finetune_seq2seq.py \
        --model_name_or_path "google/flan-t5-xl" \
        --bf16 True \
        --train_file "stanford_alpaca/alpaca_data.json" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1.0e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 5 \
        --logging_steps 10 \
        --save_steps 2000 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --output_dir ./flan-t5-xl_peft_finetuned_model \
        --peft lora
```
If you have enabled passwordless SSH in cpu clusters, you could also use mpirun in master node to start the DDP finetune. Take llama alpaca finetune for example. follow the [hugginface guide](https://huggingface.co/docs/transformers/perf_train_cpu_many) to install Intel® oneCCL Bindings for PyTorch, IPEX

oneccl_bindings_for_pytorch is installed along with the MPI tool set. Need to source the environment before using it.

for Intel® oneCCL >= 1.12.0
``` bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

for Intel® oneCCL whose version < 1.12.0
``` bash
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

The following command enables training with a total of 16 processes on 4 Xeons (node0/1/2/3, 2 sockets each node. taking node0 as the master node), ppn (processes per node) is set to 4, with two processes running per one socket. The variables OMP_NUM_THREADS/CCL_WORKER_COUNT can be tuned for optimal performance.

In node0, you need to create a configuration file which contains the IP addresses of each node (for example hostfile) and pass that configuration file path as an argument.
``` bash
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
 xxx.xxx.xxx.xxx #node2 ip
 xxx.xxx.xxx.xxx #node3 ip
```
Now, run the following command in node0 and **4DDP** will be enabled in node0 and node1 with BF16 auto mixed precision:
``` bash
export CCL_WORKER_COUNT=1
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
## for DDP ptun for LLama
mpirun -f nodefile -n 16 -ppn 4 -genv OMP_NUM_THREADS=56 python3 finetune_clm.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --train_file ./alpaca_data.json \
    --bf16 True \
    --output_dir ./llama_peft_finetuned_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --peft ptun \
    --group_by_length True \
    --dataset_concatenation \
    --use_fast_tokenizer false \
    --do_train \

## for DDP LORA for MPT
mpirun -f nodefile -n 16 -ppn 4 -genv OMP_NUM_THREADS=56 python3 finetune_clm.py \
    --model_name_or_path mosaicml/mpt-7b \
    --train_file ./alpaca_data.json \
    --bf16 True \
    --output_dir ./mpt_peft_finetuned_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4  \
    --logging_steps 1 \
    --peft lora \
    --group_by_length True \
    --dataset_concatenation \
    --do_train \
    --trust_remote_code True \
    --tokenizer_name "EleutherAI/gpt-neox-20b" \
```
you could also indicate `--peft` to switch peft method in P-tuning, Prefix tuning, Prompt tuning, LLama Adapter, LORA,
see https://github.com/huggingface/peft
